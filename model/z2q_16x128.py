import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ops import G3, FSA
import torch.nn.utils.spectral_norm as spectralnorm
from torch.nn import init


class Z2Q_16x128(nn.Module):
    def __init__(self):
        super(Z2Q_16x128, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(True)
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(2, 1), output_padding=(0, 0)),
            nn.LayerNorm(128),
            nn.LeakyReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(2, 1), output_padding=(0, 0)),
            nn.LayerNorm(128),
            nn.LeakyReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(2, 1), output_padding=(0, 0)),
            nn.LayerNorm(128),
            nn.LeakyReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(2, 1), output_padding=(0, 0)),
            nn.LayerNorm(128),
            nn.LeakyReLU(True)
        )
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.BatchNorm3d):
                init.normal_(module.weight.data, 1.0, 0.02)
                init.constant_(module.bias.data, 0.0)

    def forward(self, z_ct, z_p):
        h_cl = self.fc1(z_p)
        qfeats = torch.cat([z_ct, h_cl], dim=-1)
        qfeats = self.fc2(qfeats)
        qfeats = self.fc3(qfeats).unsqueeze(dim=1).unsqueeze(dim=1)
        word_feats = self.conv1(qfeats)
        word_feats = self.conv2(word_feats)
        word_feats = self.conv3(word_feats)
        out = self.conv4(word_feats).squeeze(dim=1)
        return out


if __name__ == '__main__':
    zc = torch.randn(2, 128)
    zp = torch.zeros(2, 128)
    zp[:, 10:30] = 1
    #za = za.repeat(1, 1, 3, 1, 1)
    net = Z2Q_16x128()
    out = net(zc, zp)
    print(out.size())
