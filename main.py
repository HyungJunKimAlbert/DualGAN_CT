import argparse
from train import *
import torch

## Parser 생성하기
parser = argparse.ArgumentParser(description="Regression Tasks such as inpainting, denoising, and super_resolution",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Set options
parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")
parser.add_argument("--task", default="DualGAN", choices=["DCGAN", "pix2pix", "cycleGAN", "DualGAN"], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['direction', 0], dest='opts')
parser.add_argument("--network", default="DualGAN", choices=["unet", "hourglass", "resnet", "srresnet", "DCGAN", "pix2pix", "cycleGAN", "DualGAN"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

# For train
parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=8, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=200, type=int, dest="num_epoch")
parser.add_argument("--img_size", default=512, type=int, dest="img_size")
parser.add_argument("--in_channels", default=1, type=int, dest="in_channels")
parser.add_argument("--wgt_cycle", default=10, type=float, dest="wgt_cycle")
parser.add_argument("--wgt_adv", default=1, type=float, dest="wgt_adv")
parser.add_argument("--wgt_gp", default=10, type=float, dest="wgt_gp")

# For results
parser.add_argument("--data_dir", default="/home/hjkim/projects/local_dev/dental_cycleGAN/data", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

args = parser.parse_args()


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["TORCH_USE_CUDA_DSA"] = '1'
    
    if args.mode == "train":
        train(args)
    # if args.mode == "test":
    #     test(args)
