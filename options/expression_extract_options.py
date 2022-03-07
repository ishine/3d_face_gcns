import argparse
import torch


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_dir', type=str, default=None)
        self.parser.add_argument('--dataset_mode', type=str, default='irls')
        self.parser.add_argument('--matlab_data_path', type=str, default='renderer/data/data.mat')

        self.parser.add_argument('--batch_size', type=int, default=10)
        self.parser.add_argument('--serial_batches', type=self.str2bool, default=True)
        self.parser.add_argument('--drop_last', type=self.str2bool, default=False)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--isTrain', type=self.str2bool, default=True)
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self.parser.add_argument('--lambda_photo', type=float, default=1.0)
        self.parser.add_argument('--lambda_land', type=float, default=10)
        self.parser.add_argument('--lambda_reg', type=float, default=2.5e-5)

        self.parser.add_argument('--lambda_alpha', type=float, default=1.0)
        self.parser.add_argument('--lambda_beta', type=float, default=1.0)
        self.parser.add_argument('--lambda_delta', type=float, default=50)

        self.parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')

        self.parser.add_argument('--image_width', type=int, default=256)
        self.parser.add_argument('--image_height', type=int, default=256)


    def parse_args(self):
        self.args = self.parser.parse_args()
        self.args.device = torch.device('cuda:{}'.format(self.args.gpu_ids[0])) if self.args.gpu_ids else torch.device('cpu')
        return self.args
    
    def test_parse_args(self):
        self.args = self.parser.parse_args("")
        self.args.device = torch.device('cuda:{}'.format(self.args.gpu_ids[0])) if self.args.gpu_ids else torch.device('cpu')
        return self.args

    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
