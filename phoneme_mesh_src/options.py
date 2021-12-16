import argparse
import torch


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_dir', type=str, default=None)
        self.parser.add_argument('--src_dir', type=str, default=None)
        self.parser.add_argument('--tgt_dir', type=str, default=None)

        self.parser.add_argument('--train_rate', type=float, default=0.8)
        self.parser.add_argument('--num_epoch', type=int, default=500)
        self.parser.add_argument('--batch_size', type=int, default=128)
        self.parser.add_argument('--serial_batches', type=self.str2bool, default=False)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--isTrain', type=self.str2bool, default=True)
        self.parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate for adam')
        self.parser.add_argument('--lambda_geo', type=float, default=0.3)
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self.parser.add_argument('--display_port', type=int, default=11111, help='tensorboard port of the web display')
        self.parser.add_argument('--display_freq', type=int, default=2000, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        self.parser.add_argument('--freeze_mesh', type=bool, default=False, help='Choose if you want to freeze mesh training pipeline or not')
        self.parser.add_argument('--load_only_mesh', type=bool, default=False, help='Load only Audio Encoder and Mesh Decoder part from the save file')
        self.parser.add_argument('--load_model', type=bool, default=False, help='Load model from the checkpoint')
        self.parser.add_argument('--model_name', type=str, default=False, help='Name of the checkpoint file')
        self.parser.add_argument('--save_name', type=str, default=None)
        self.parser.add_argument('--use_adam', action='store_true')
        self.parser.add_argument('--use_adabound', action='store_true')
        self.parser.add_argument('--adabound_amsbound', action='store_true')
        self.parser.add_argument('--use_texture', action='store_true')

        self.parser.add_argument('--no_audioenc', action='store_true')
        self.parser.add_argument('--use_mel_target', action='store_true')
        self.parser.add_argument('--use_src_seq', action='store_true')
        self.parser.add_argument('--use_enc_pre', action='store_true')
        self.parser.add_argument('--dense_enc', action='store_true')
        

    def parse_args(self):
        self.args = self.parser.parse_args()
        if self.args.gpu_ids == "cpu":
            self.args.device = torch.device("cpu")
        else:
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
