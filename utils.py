import torch
from torch import nn
from torch.nn import functional as F

from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from torch.optim import lr_scheduler

from dataload import load_cifar_10, load_cifar_100, load_tiny_imagenet
from network import ResNet20, Vgg, ResNet18
from pruner import Pruner
from train import train_net


def set_axes(axes, xlabel, ylabel, xlim, ylim, legend, xscale, yscale):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, title=None,
         ylim=None, xscale='linear', yscale='linear', xticklabels=None,
         fmts=('^-r', '-b', '-g', '-m', '-C1', '-C5'), figsize=(4.5, 4), axes=None, twins=False, ylim2=None):
    backend_inline.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    if twins:
        ax2 = axes.twinx()
        ax2.set_ylim(ylim2)
        ax2.set_ylabel(ylabel[1])
    i = 0
    ax = axes
    f = []
    for x, y, fmt in zip(X, Y, fmts):
        if twins and (i > 0):
            ax = ax2
        if len(x):
            h, = ax.plot(x, y, fmt)
        else:
            h, = ax.plot(y, fmt)
        f.append(h)
        i += 1
    if title:
        axes.set_title(title)
    set_axes(axes, xlabel, ylabel[0], xlim, ylim, legend, xscale, yscale)
    if xticklabels:
        axes.set_xticks(X[0])
        axes.set_xticklabels(xticklabels, rotation=60, fontsize=12)


class Tester:
    @staticmethod
    def test_masked_resnet20(final_s, path="models/res", T=20, e=1.e-2):
        epoch, batch_size, lr = 160, 128, 0.1
        num_classes, prune_batch_size = 10, 256

        loss = nn.CrossEntropyLoss().cuda(torch.device("cuda:0"))

        data_iter = load_cifar_10(batch_size)

        net = ResNet20().cuda(torch.device("cuda:0"))
        resnet_pruner = Pruner(e, final_s)
        resnet_pruner.prune(net, T, 5, (num_classes * 10, 3, 32, 32), prune_batch_size)

        trainer = torch.optim.SGD(net.parameters(), weight_decay=1e-4, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[80, 120], gamma=0.1)
        sparsity_ratio, test_acc = train_net(net, loss, trainer, data_iter, epoch, path, scheduler)
        print("true sparsity: %f%%    test_acc: %.2f%%" % (sparsity_ratio * 100, test_acc * 100))
        return sparsity_ratio, test_acc

    @staticmethod
    def test_masked_resnet20_muti(path="models/res"):
        compress_list = torch.arange(2, 20, 2)
        sparsity = 1 - 0.8 ** compress_list

        sparsity_ratios, test_accs = [], []
        for final_s in sparsity:
            sparsity_ratio, test_acc = Tester.test_masked_resnet20(final_s, path)
            sparsity_ratios.append(sparsity_ratio)
            test_accs.append(test_acc)

        plot(compress_list, torch.tensor(test_accs) * 100, ylim=[76, 92],
             xlabel='Sparsity (%)', ylabel=['Test Accuracy (%)'], title="ResNet-20 (CIFAR-10)",
             xticklabels=[str(round(s * 100, 2)) for s in sparsity.tolist()], figsize=(6, 5))
        plt.savefig("res/ResNet-20 (CIFAR-10)")

    @staticmethod
    def test_masked_vgg16(final_s, path="models/vgg16"):
        epoch, batch_size, lr = 160, 128, 0.1
        T, num_classes, prune_batch_size = 100, 100, 256

        loss = nn.CrossEntropyLoss().cuda(torch.device("cuda:0"))

        data_iter = load_cifar_100(batch_size)

        net = Vgg(num_classes=num_classes).cuda(torch.device("cuda:0"))
        resnet_pruner = Pruner(1.e-2, final_s)
        resnet_pruner.prune(net, T, 5, (num_classes * 10, 3, 32, 32), prune_batch_size)

        trainer = torch.optim.SGD(net.parameters(), weight_decay=5e-4, lr=lr, momentum=0.9, nesterov=True)
        scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[60, 120], gamma=0.1)
        sparsity_ratio, test_acc = train_net(net, loss, trainer, data_iter, epoch, path, scheduler)
        print("true sparsity: %f%%    test_acc: %.2f%%" % (sparsity_ratio * 100, test_acc * 100))
        return sparsity_ratio, test_acc

    @staticmethod
    def test_masked_vgg16_muti(path="models/vgg"):
        compress_list = torch.arange(2, 20, 2)
        sparsity = 1 - 0.8 ** compress_list

        sparsity_ratios, test_accs = [], []
        for final_s in sparsity:
            sparsity_ratio, test_acc = Tester.test_masked_vgg16(final_s, path)
            sparsity_ratios.append(sparsity_ratio)
            test_accs.append(test_acc)

        plot(compress_list, torch.tensor(test_accs) * 100, #ylim=[66, 75],
             xlabel='Sparsity (%)', ylabel=['Test Accuracy (%)'], title="VGG-16 (CIFAR-100)",
             xticklabels=[str(round(s * 100, 2)) for s in sparsity.tolist()], figsize=(8, 8))
        plt.savefig("res/VGG-16 (CIFAR-100)")

    @staticmethod
    def test_masked_resnet18(final_s, path="models/res18"):
        epoch, batch_size, lr = 200, 256, 0.2
        T, num_classes, prune_batch_size = 100, 200, 128  # 256太大

        loss = nn.CrossEntropyLoss().cuda(torch.device("cuda:0"))

        data_iter = load_tiny_imagenet(batch_size)

        net = ResNet18().cuda(torch.device("cuda:0"))
        resnet_pruner = Pruner(1.e-2, final_s)
        resnet_pruner.prune(net, T, 5, (num_classes * 10, 3, 64, 64), prune_batch_size)

        trainer = torch.optim.SGD(net.parameters(), weight_decay=1e-4, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[100, 150], gamma=0.1)
        sparsity_ratio, test_acc = train_net(net, loss, trainer, data_iter, epoch, path, scheduler)
        print("true sparsity: %f%%    test_acc: %.2f%%" % (sparsity_ratio * 100, test_acc * 100))
        return sparsity_ratio, test_acc

    @staticmethod
    def test_masked_resnet18_muti(path="models/resnet18"):
        compress_list = torch.arange(2, 20, 2)
        sparsity = 1 - 0.8 ** compress_list

        sparsity_ratios, test_accs = [], []
        for final_s in sparsity:
            sparsity_ratio, test_acc = Tester.test_masked_resnet18(final_s, path)
            sparsity_ratios.append(sparsity_ratio)
            test_accs.append(test_acc)

        plot(compress_list, torch.tensor(test_accs) * 100,  # ylim=[66, 75],
             xlabel='Sparsity (%)', ylabel=['Test Accuracy (%)'], title="res/ResNet-18 (Tiny-ImageNet)",
             xticklabels=[str(round(s * 100, 2)) for s in sparsity.tolist()], figsize=(8, 8))
        plt.savefig("res/ResNet-18(Tiny-ImageNet)")

    @staticmethod
    def compute_NTK():
        compress_list = torch.arange(2, 20, 2)
        sparsity = 1 - 0.8 ** compress_list

        T, num_classes, prune_batch_size = 20, 10, 256

        for final_s in sparsity:
            net = ResNet20().cuda(torch.device("cuda:0"))
            resnet_pruner = Pruner(1.e-2, final_s)
            dense_eigenvalues, sparse_eigenvalues = resnet_pruner.prune(net, T, 5, (num_classes * 10, 3, 32, 32),
                                                                        prune_batch_size, NTK_show=True)
            size = dense_eigenvalues.shape[0]
            plot(torch.arange(size)+1, [dense_eigenvalues, sparse_eigenvalues],
                 legend=['Dense', 'NTK-SAP'], yscale='log', fmts=['-C7', '-r'],
                 xlabel='Eigenvalue index', ylabel=['Magnitude of each eigenvalue'], figsize=(8, 8))
            plt.savefig(f"res/NTK_{int(final_s*100)}.png")

            plot(torch.arange(size) + 1, [dense_eigenvalues.sort()[0], sparse_eigenvalues.sort()[0]],
                 legend=['Dense', 'NTK-SAP'], yscale='log', fmts=['-C7', '-r'],
                 xlabel='Eigenvalue index', ylabel=['Magnitude of each eigenvalue'], figsize=(8, 8))
            plt.savefig(f"res/NTK_{int(final_s*100)}_sort.png")

    @staticmethod
    def test_masked_resnet20_muti_T(path="models/res"):
        compress_list = torch.arange(2, 20, 2)
        sparsity = 1 - 0.8 ** compress_list
        Ts = [1, 2, 4, 10, 20]

        test_accs_T = []
        for T in Ts:
            test_accs = []
            for final_s in sparsity:
                _, test_acc = Tester.test_masked_resnet20(final_s, path, T=T)
                test_accs.append(test_acc)
            test_accs_T.append(test_accs)

        plot(compress_list, torch.tensor(test_accs_T) * 100, xlabel='Sparsity (%)',
             xticklabels=[str(round(s * 100, 2)) for s in sparsity.tolist()], figsize=(8, 8),
             fmts=['o-m', 'o-y', 'o-g', 'o-c', 'o-b'],
             ylabel=['Test Accuracy (%)'], legend=['T=1', 'T=2', 'T=4', 'T=10', 'T=20'])
        plt.savefig("res/ResNet-20(CIFAR-10) muti_T")

    @staticmethod
    def test_masked_resnet20_muti_e(path="models/res"):
        compress_list = torch.arange(2, 20, 2)
        sparsity = 1 - 0.8 ** compress_list
        epsilons = torch.sqrt(torch.tensor([1e-4, 2.5e-5, 1e-5, 1e-6]))

        test_accs_T = []
        for i, e in enumerate(epsilons):
            test_accs = []
            for final_s in sparsity:
                _, test_acc = Tester.test_masked_resnet20(final_s, path, e=e)
                test_accs.append(test_acc)
            test_accs_T.append(test_accs)

        plot(compress_list, torch.tensor(test_accs_T) * 100, xlabel='Sparsity (%)',
             xticklabels=[str(round(s * 100, 2)) for s in sparsity.tolist()], figsize=(8, 8),
             fmts=['o-m', 'o-y', 'o-g', 'o-c'],
             ylabel=['Test Accuracy (%)'], legend=['ϵ=1e-4', 'ϵ=2.5e-5', 'ϵ=1e-5', 'ϵ=1e-6'])
        plt.savefig("res/ResNet-20(CIFAR-10) muti_e")
