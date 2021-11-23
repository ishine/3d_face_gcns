import argparse
import torch


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_dir', type=str, default=None)
        self.parser.add_argument('--src_dir', type=str, default=None)
        self.parser.add_argument('--tgt_dir', type=str, default=None)
        self.parser.add_argument('--bfm_dir', type=str, default=None)
        self.parser.add_argument('--mesh_dir', type=str, default=None)
        self.parser.add_argument('--matlab_data_path', type=str, default='renderer/data/data.mat')
        self.parser.add_argument('--pretrained_model_path', type=str, default='weights/resnet50_ft_weight.pkl')
        
        self.parser.add_argument('--lambda_delta', type=float, default=1.0)
        self.parser.add_argument('--lambda_land', type=float, default=1.6e-3)
        self.parser.add_argument('--lambda_photo', type=float, default=0)
        self.parser.add_argument('--lambda_reg', type=float, default=3e-4)
        self.parser.add_argument('--lambda_geo', type=float, default=1.0)  # 5e-2
        self.parser.add_argument('--lambda_face_emb', type=float, default=1e-2)
        self.parser.add_argument('--lambda_media', type=float, default=0)
        self.parser.add_argument('--lambda_temporal', type=float, default=1)
        self.parser.add_argument('--lambda_emotion', type=float, default=0.1)

        self.parser.add_argument('--train_rate', type=float, default=0.8)
        self.parser.add_argument('--num_epoch', type=int, default=500)
        self.parser.add_argument('--batch_size', type=int, default=128)
        self.parser.add_argument('--serial_batches', type=self.str2bool, default=False)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--isTrain', type=self.str2bool, default=True)
        self.parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate for adam')
        self.parser.add_argument('--lr_decay', type=float, default=0.9)
        self.parser.add_argument('--lr_decay_step', type=float, default=2)

        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self.parser.add_argument('--display_port', type=int, default=11111, help='tensorboard port of the web display')
        self.parser.add_argument('--display_freq', type=int, default=2000, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        self.parser.add_argument('--suffix', type=str, default=None)

    def parse_args(self):
        self.args = self.parser.parse_args()
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
