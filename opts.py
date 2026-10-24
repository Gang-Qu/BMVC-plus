import argparse

## /home/qugang/EfficientSPI/weights/E2E-DUNet/2025_05_16_14_48_21/epoch_63.pth  dual path with mamba and attention
## /qugang/EfficientSPI/EfficientSPI/weights/E2E-DUNet/2025_05_27_17_50_07/epoch_5.pth EfficientSCI++
## /qugang/EfficientSPI/EfficientSPI/weights/E2E-DUNet/2025_05_28_16_05_47/epoch_85.pth  cascaded and 3 stages and 128 channels
## /qugang/EfficientSPI/EfficientSPI/weights/E2E-DUNet/2025_05_29_13_47_32/epoch_93.pth  4,16
## /qugang/EfficientSPI/EfficientSPI/weights/E2E-DUNet/2025_05_29_18_56_53/epoch_95.pth  9,8
## /qugang/EfficientSPI/EfficientSPI/weights/E2E-DUNet/2025_05_30_01_30_56/epoch_88.pth  10, 10
## /qugang/EfficientSPI/EfficientSPI/weights/E2E-DUNet/2025_06_02_14_09_32/epoch_93.pth  1S+abc weight


##0920
## /mnt/inaisfs/home/yuanxin/qugang/EfficientSPI/EfficientSPI/weights/A-0920_E2E-DUNet/2025_09_20_13_43_57/epoch_40.pth  mamba  29.24/0.7930

## /mnt/inaisfs/home/yuanxin/qugang/EfficientSPI/EfficientSPI/weights/A-0920_E2E-DUNet/2025_09_21_13_56_40/epoch_10.pth  SwinTrans break  28.36/0.7598
## /mnt/inaisfs/home/yuanxin/qugang/EfficientSPI/EfficientSPI/weights/A-0920_E2E-DUNet/2025_09_22_01_38_52/epoch_63.pth  from 10_SwinTrans  29.23/0.7928
## /mnt/inaisfs/home/yuanxin/qugang/EfficientSPI/EfficientSPI/weights/A-0920_E2E-DUNet/2025_09_25_18_51_50/epoch_99.pth  from 63 SwinTrans  29.33/0.7951

## /mnt/inaisfs/home/yuanxin/qugang/EfficientSPI/EfficientSPI/weights/A-0920_E2E-DUNet/2025_09_24_17_11_20/epoch_75.pth //from 40 4s_mamba 29,33/0.7951 cascaded CA+mamba+ffn

## /mnt/inaisfs/home/yuanxin/qugang/EfficientSPI/EfficientSPI/weights/A-0920_E2E-DUNet/2025_09_27_00_35_12/epoch_185.pth  only ffn  29.28/0.7940
## /mnt/inaisfs/home/yuanxin/qugang/EfficientSPI/EfficientSPI/weights/A-0920_E2E-DUNet/2025_09_26_11_57_35/epoch_100.pth  mb+ffn    29.36/0.7958
## /mnt/inaisfs/home/yuanxin/qugang/EfficientSPI/EfficientSPI/weights/A-0920_E2E-DUNet/2025_09_25_13_57_36/epoch_71.pth   cn+ffn    29.18/0.7920
## /mnt/inaisfs/home/yuanxin/qugang/EfficientSPI/EfficientSPI/weights/A-0920_E2E-DUNet/2025_09_27_11_45_36/epoch_139.pth  mb+cn+ffn 29.40/0.7966
## /mnt/inaisfs/home/yuanxin/qugang/EfficientSPI/EfficientSPI/weights/A-0920_E2E-DUNet/2025_09_29_01_52_37/epoch_153.pth  mb+cn+ffn  lr=0.00005  29.42/0.7972

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2000, type=int)
    # parser.add_argument("--test_weight_path", default=None, type=str, help='testing on benchmark')
    parser.add_argument("--test_weight_path", default='/qugang/EfficientSPI/EfficientSPI/weights/A-0920_E2E-DUNet/2025_09_20_13_43_57/epoch_40.pth', type=str, help='testing on benchmark')
    parser.add_argument('--pretrained_model_path', default='/qugang/EfficientSPI/EfficientSPI/weights/A-0920_E2E-DUNet/2025_09_27_11_45_36/epoch_139.pth', type=str, help='pretrained model path')
    # parser.add_argument('--pretrained_model_path', default=None, type=str, help='pretrained model path')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--decoder_type', default='A-0920_E2E-DUNet', type=str, help='reconstruction network type')
    parser.add_argument('--cr', default=[4, 8], type=float)
    parser.add_argument('--size', default=[1080, 1920])
    parser.add_argument('--meas_size', default=[272, 240]) ## note: need to cinsider padding, when H or W cannot be devided by 2^2. e.g.[270, 240]--->[272, 240]
    parser.add_argument('--matrix_train', default=True, type=bool)
    parser.add_argument('--lr', default=0.00005, type=float)
    parser.add_argument('--stages', default=3, type=int)
    parser.add_argument('--channels', default=128, type=int)
    parser.add_argument('--mid_blocks', default=1)
    parser.add_argument('--enc_blocks', default=[1, 1])
    parser.add_argument('--dec_blocks', default=[1, 1])
    parser.add_argument('--save_model_step', default=1, type=int)
    parser.add_argument('--save_train_image_step', default=500, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--iter_step', default=100, type=int)
    parser.add_argument('--test_flag', default=True, type=bool)
    parser.add_argument('--train_data_path', type=str, default='/qugang/BMVC+/Datasets/train_data_2/HR_20230221')
    # parser.add_argument('--train_data_path', type=str, default='/qugang/EfficientSPI/test_from_davis_1920_1080/test_from_davis_1920_1080')
    # parser.add_argument('--train_data_path', type=str, default='./dataset/Set11')
    parser.add_argument('--benchmark_path', type=str, default='/qugang/EfficientSPI/test_from_davis_1920_1080/test_from_davis_1920_1080')
    # parser.add_argument('--benchmark_path', type=str, default='/qugang/BMVC+/Datasets/test_data/test_from_davis')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                        help="Enable compilation w/ specified backend (default: inductor).")
    args = parser.parse_args()

    return args
