import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    from utils import Tester

    # Tester.test_masked_resnet20(0.95, path=sys.argv[2])
    # Tester.test_masked_resnet20_muti(path=sys.argv[2])
    # Tester.test_masked_vgg16(0.95, path=sys.argv[2])
    # Tester.test_masked_vgg16_muti(path=sys.argv[2])
    # Tester.test_masked_resnet18(0.956, path=sys.argv[2])
    # Tester.test_masked_resnet18_muti(path=sys.argv[2])
    # Tester.compute_NTK()
    # Tester.test_masked_resnet20_muti_T(path=sys.argv[2])
    Tester.test_masked_resnet20_muti_e(path=sys.argv[2])
