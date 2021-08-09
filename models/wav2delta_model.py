import torch
from torch import nn
from conv import Conv2d


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.zeros_(m.bias)


class Wav2DeltaModel(nn.Module):
    def __init__(self, output_dim=64):
        super(Wav2DeltaModel, self).__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)
            
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        pretrained_audio_encoder_path = "models/wav2delta_pretrained.pth"
        self.init_weights(pretrained_audio_encoder_path)

    
    def init_weights(self, pretrained_model_path):
        ckpt = torch.load(pretrained_model_path, map_location='cpu')

        own_state = self.state_dict()
        pretrained_dict = {k: v for k, v in ckpt.items() if k in own_state}
        own_state.update(pretrained_dict)

        self.load_state_dict(own_state)

        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        self.fc.apply(weights_init)


    def forward(self, x):
        # x = (B, T, 1, 80, 16)
        B = x.size(0)
        input_dim_size = len(x.size())
        if input_dim_size > 4:
            x = torch.cat([x[:, i] for i in range(x.size(1))], dim=0)
        out = self.audio_encoder(x).flatten(start_dim=1)
        out = self.fc(out)

        if input_dim_size > 4:
            out = torch.split(out, B, dim=0)  # [(B, 64) * T]
            out = torch.stack(out, dim=1)
        return out
