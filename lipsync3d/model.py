import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)


class Lipsync3DModel(nn.Module):
    def __init__(self, use_auto_regressive=False):
        super(Lipsync3DModel, self).__init__()
        self.use_auto_regressive=use_auto_regressive

        self.AudioEncoder = nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=72, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),   # 2 x 256 x 24 -> 72 x 128 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=72,out_channels=108, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 72 x 128 x 24 -> 108 x 64 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=108,out_channels=162, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 108 x 64 x 24 -> 162 x 32 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=162,out_channels=243, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 162 x 32 x 24 -> 243 x 16 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=243,out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 243 x 16 x 24 -> 256 x 8 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 256 x 8 x 24 -> 256 x 4 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256,out_channels=128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # 256 x 4 x 24 -> 128 x 4 x 13
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128,out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # 128 x 4 x 13 -> 64 x 4 x 8
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # 64 x 4 x 8 -> 32 x 4 x 5
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # 32 x 4 x 5 -> 16 x 4 x 4
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16,out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # 16 x 4 x 4 -> 8 x 4 x 3
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8,out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), dilation=1),     # 8 x 4 x 3 -> 4 x 4 x 2
            nn.LeakyReLU(),
            View([-1, 32]),
        )

        self.GeometryDecoder = nn.Sequential(
            nn.Linear(32, 150),
            nn.Dropout(0.5),
            nn.Linear(150, 1434)
        )

        self.TextureEncoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=1024, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024,out_channels=2048, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*4*2048, 2),
            nn.Tanh()
        )

        if self.use_auto_regressive:
            input_dim = 34
        else:
            input_dim = 32
        
        self.TextureDecoder = nn.Sequential(
            nn.Linear(input_dim, 4*4*1024),
            nn.ReLU(),
            View([-1, 1024, 4, 4]),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Tanh(),
        )

    def forward(self, spec, texture_pred=None):
        # spec : B x 2 x 256 x 24
        # texture : B x 3 x 128 x 128

        latent = self.AudioEncoder(spec)
        geometry_diff = self.GeometryDecoder(latent)
        # if self.use_auto_regressive:
        #     texture_latent = self.TextureEncoder(texture_pred)
        #     latent = torch.cat([latent, texture_latent], 1)
        # texture_diff = self.TextureDecoder(latent)

        # return geometry_diff, texture_diff
        return geometry_diff