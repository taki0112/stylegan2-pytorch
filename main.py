import argparse
from utils import *
from StyleGAN2 import run_fn

def parse_args():
    desc = "Pytorch implementation of StyleGAN2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train, test, draw]')
    parser.add_argument('--dataset', type=str, default='FFHQ', help='dataset_name')
    parser.add_argument('--nsml', type=str2bool, default=False, help='NAVER NSML use or not')

    parser.add_argument('--n_total_image', type=int, default=25000, help='The total iterations')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--batch_size', type=int, default=8, help='batch sizes for each gpus')

    parser.add_argument('--lazy_regularization', type=str2bool, default=True, help='lazy_regularization')
    parser.add_argument('--d_reg_every', type=int, default=16, help='interval of the applying r1 regularization')
    parser.add_argument('--g_reg_every', type=int, default=4, help='interval of the applying path length regularization')
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument('--channel_multiplier', type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--mixing_prob", type=float, default=0.9, help="probability of latent code mixing")


    parser.add_argument('--print_freq', type=int, default=2000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of ckpt_save_freq')
    parser.add_argument('--n_sample', type=int, default=64, help='number of the samples generated during training')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one', flush=True)

    return args

"""main"""
def main():

    args = vars(parse_args())

    # run
    multi_gpu_run(ddp_fn=run_fn, args=args)



if __name__ == '__main__':
    main()