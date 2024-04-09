import copy
import types

import torch
from torch import nn
from torch.nn import functional as F

from dataload import load_prune_data


def reset_BN(net):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.reset_running_stats()


def set_BN(net1, net2):
    for module, module_mod in zip(net1.modules(), net2.modules()):
        if isinstance(module, nn.BatchNorm2d):
            module_mod.running_mean = module.running_mean
            module_mod.running_var = module.running_var
            module_mod.num_batches_tracked = module.num_batches_tracked

class Pruner:
    def __init__(self, epsilon, final_s):
        self.epsilon = epsilon
        self.final_s = final_s
        self.weight_num = 0
        self.masks = {}
        self.parameters = {}
        self.perturb_parameters = {}

    def init_parameters(self):
        for name, weight in self.parameters.items():
            if weight.ndim == 1:
                weight.data.fill_(1.)
            else:
                nn.init.kaiming_normal_(weight)
            weight.detach_().mul_(self.masks[name])

        for name, weight in self.perturb_parameters.items():
            if weight.ndim == 1:
                weight.data.fill_(1.)
            else:
                weight.data = self.parameters[name].data + torch.randn(*weight.shape, device=weight.device) * self.epsilon
            weight.detach_().mul_(self.masks[name])

    def compute_score(self, T, R, data_iter, net1, net2, device):
        # momentum 设为1，为每个 batch 的数据独立计算均值和方差
        for module_1, module_2 in zip(net1.modules(), net2.modules()):
            if isinstance(module_1, nn.BatchNorm2d):
                module_1.momentum = 1.0
                module_2.momentum = 1.0

        for t in range(T):  # 剪枝 T 次，逐次达到最终稀疏度
            for _ in range(R):  # 采样 R 次
                for data in data_iter:  # NINP
                    self.init_parameters()
                    X = torch.randn_like(data, device=device)

                    reset_BN(net1)
                    reset_BN(net2)
                    net1.train()
                    net2.train()
                    with torch.no_grad():  # 为每个 batch 的数据独立计算均值和方差
                        net1(X)
                        set_BN(net1, net2)
                    net1.eval()
                    net2.eval()

                    y1 = net1(X)
                    y2 = net2(X)
                    term = torch.sum((y2 - y1) ** 2)
                    term.backward()

            # 已经被遮盖的位置不在考虑其梯度，即不会再连接
            masks_grad = [(mask.grad * (mask.detach() != 0)).view(-1) for mask in self.masks.values()]
            score = torch.abs(torch.cat(masks_grad))
            keep_weight_num = int((1 - self.final_s) ** ((t + 1) / T) * self.weight_num)
            threshold = torch.topk(score, keep_weight_num)[0][-1]
            with torch.no_grad():
                cnt = 0
                for mask in self.masks.values():
                    p = torch.abs(mask.grad) * (mask.detach() != 0)
                    mask[:] = (p >= threshold).float()
                    mask.grad.data.zero_()
                    cnt += torch.sum(mask != 0)
                # print(t, threshold, keep_weight_num, cnt)

    @staticmethod
    def apply_mask(module, mask):
        mask.require_grad = False
        nn.init.kaiming_normal_(module.weight)
        module.weight.data *= mask
        module.weight.register_hook(lambda grad: grad * mask)  # 冻结梯度

    def prune(self, net, T, R, size, batch_size, NTK_show=False):
        device = next(net.parameters()).device

        net1 = copy.deepcopy(net)
        for name, module in net1.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
                module.weight.requires_grad = False
                self.masks[name] = torch.ones_like(module.weight, requires_grad=True, device=device)
                self.parameters[name] = module.weight
                self.weight_num += module.weight.numel()

        net2 = copy.deepcopy(net1)
        for name, module in net2.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
                self.perturb_parameters[name] = module.weight

        data_iter = load_prune_data(size, batch_size, device)
        self.compute_score(T, R, data_iter, net1, net2, device)

        if NTK_show:
            dense_eigenvalues = self.compute_NTK(net, size, device)

        for name, module in net.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                self.apply_mask(module, self.masks[name])
            if isinstance(module, nn.BatchNorm2d):  # Reset momentum of BatchNorm2d
                module.momentum = 0.1

        del net1, net2

        if NTK_show:
            sparse_eigenvalues = self.compute_NTK(net, size, device, reinit=False)
            return dense_eigenvalues, sparse_eigenvalues

    def compute_NTK(self, net, size, device, reinit=True):
        parameters = {}
        for name, module in net.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                parameters[name] = module.weight
                if reinit:
                    nn.init.kaiming_normal_(module.weight)

        res = []
        X = torch.randn(*size, device=device)
        num_classes = size[0] // 10
        for x in X:
            y_hat = net(torch.unsqueeze(x, dim=0)).squeeze(dim=0)
            for i in range(num_classes):
                y_hat[i].backward(retain_graph=(i < num_classes-1))
                cur = []
                for weight in parameters.values():
                    cur.append(torch.clone(weight.grad).view(-1))
                    weight.grad.data.zero_()
                res.append(torch.cat(cur))

        P = torch.stack(res)
        eigenvalues, _ = torch.linalg.eig(P @ P.T)
        return torch.abs(eigenvalues).detach().cpu()
