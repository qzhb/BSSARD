import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ops import G3, FSA
import torch.nn.utils.spectral_norm as spectralnorm
from torch.nn import init

class Z128x128(nn.Module):
    def __init__(self, c_a=128, c_m=10, ch=64, mode='1p2d', use_attention=True):
        super(Z128x128, self).__init__()

        self.use_attention = use_attention

        self.block1 = G3(c_a, c_a + c_m, c_m, ch * 8, mode, 2, 5, 1, 1, 0, 0)  #
        self.block2 = G3(ch * 8, ch * 8 * 2, ch * 8, ch * 8, mode, 1, 4, 1, 2, 0, 1)  #
        self.block3 = G3(ch * 8, ch * 8 * 2, ch * 8, ch * 4, mode, 1, 4, 1, 4, 0, 0)  #
        self.block4 = G3(ch * 4, ch * 4 * 2, ch * 4, ch * 2, mode, 1, 4, 1, 2, 0, 1)  #
        self.block5 = G3(ch * 2, ch * 4 + 3, ch * 2 + 3, ch * 2, mode, 1, 1, 1, 1, 0, 0)  #
        self.block6 = G3(ch * 2, ch * 4, ch * 2, ch, mode, 1, 1, 1, 1, 0, 0)  #

        self.fsa6 = FSA(ch * 2)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.BatchNorm3d):
                init.normal_(module.weight.data, 1.0, 0.02)
                init.constant_(module.bias.data, 0.0)

    def forward(self, za, zm, zp, query_features=None):

        zp = zp.transpose(1, 2)
        # za:[1, 128, 1, 1, 1]
        # za = za.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # zm:[1, 10, 4, 1, 1]
        # zm = zm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # zv:[1, 138, 4, 1, 1]
        zv = torch.cat([za.repeat(1, 1, zm.size(2), 1, 1), zm], 1)
        # hs:[1, 512, 1, 4, 4];  hv:[1, 1024, 7, 4, 4];  ht:[1, 512, 7, 1, 1]
        hs, hv, ht = self.block1(za, zv, zm)
        # hs:[1, 512, 1, 8, 8];  hv:[1, 1024, 14, 8, 8];  ht:[1, 512, 14, 1, 1]
        hs, hv, ht = self.block2(hs, hv, ht)
        # hs:[1, 256, 1, 16, 16];  hv:[1, 512, 28, 16, 16];  ht:[1, 256, 28, 1, 1]
        hs, hv, ht = self.block3(hs, hv, ht)
        # hs:[1, 128, 1, 32, 32];  hv:[1, 256, 28, 32, 32];  ht:[1, 128, 28, 1, 1]
        hs, hv, ht = self.block4(hs, hv, ht)

        hv = F.avg_pool3d(hv, kernel_size=(1, 2, 2)).squeeze()
        hv = torch.cat([hv, zp], 1).unsqueeze(-1).unsqueeze(-1)

        hs = F.avg_pool3d(hs, kernel_size=(1, 2, 2))

        ht = ht.squeeze()
        ht = torch.cat([ht, zp], 1).unsqueeze(-1).unsqueeze(-1)

        hs, hv, ht = self.block5(hs, hv, ht)
        hs, hv, ht = self.block6(hs, hv, ht)
        hv = self.fsa6(hv)
        out = hv.squeeze().transpose(1, 2)

        return out


class Z128x128_2(nn.Module):
    def __init__(self, c_a=128, c_m=10, ch=64, mode='1p2d', use_attention=True):
        super(Z128x128_2, self).__init__()

        self.use_attention = use_attention

        self.block1 = G3(c_a, c_a + c_m, c_m, ch * 8, mode, 2, 5, 1, 1, 0, 0)  #
        self.block2 = G3(ch * 8, ch * 8 * 2, ch * 8, ch * 8, mode, 1, 4, 1, 2, 0, 1)  #
        self.block3 = G3(ch * 8, ch * 8 * 2, ch * 8, ch * 4, mode, 1, 4, 1, 4, 0, 0)  #
        self.block4 = G3(ch * 4, ch * 4 * 2, ch * 4, ch * 2, mode, 1, 4, 1, 2, 0, 1)  #
        self.block5 = G3(ch * 2, ch * 4 + 1, ch * 2 + 1, ch * 2, mode, 1, 1, 1, 1, 0, 0)  #
        self.block6 = G3(ch * 2, ch * 4, ch * 2, ch, mode, 1, 1, 1, 1, 0, 0)  #

        self.fsa6 = FSA(ch * 2)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.BatchNorm3d):
                init.normal_(module.weight.data, 1.0, 0.02)
                init.constant_(module.bias.data, 0.0)

    def forward(self, za, zm, zp, query_features=None):
        zp = zp.argmax(dim = 2).unsqueeze(1)
        # zv:[1, 138, 4, 1, 1]
        zv = torch.cat([za.repeat(1, 1, zm.size(2), 1, 1), zm], 1)
        # hs:[1, 512, 1, 4, 4];  hv:[1, 1024, 7, 4, 4];  ht:[1, 512, 7, 1, 1]
        hs, hv, ht = self.block1(za, zv, zm)
        # hs:[1, 512, 1, 8, 8];  hv:[1, 1024, 14, 8, 8];  ht:[1, 512, 14, 1, 1]
        hs, hv, ht = self.block2(hs, hv, ht)
        # hs:[1, 256, 1, 16, 16];  hv:[1, 512, 28, 16, 16];  ht:[1, 256, 28, 1, 1]
        hs, hv, ht = self.block3(hs, hv, ht)
        # hs:[1, 128, 1, 32, 32];  hv:[1, 256, 28, 32, 32];  ht:[1, 128, 28, 1, 1]
        hs, hv, ht = self.block4(hs, hv, ht)

        hv = F.avg_pool3d(hv, kernel_size=(1, 2, 2)).squeeze()
        hv = torch.cat([hv, zp], 1).unsqueeze(-1).unsqueeze(-1)

        hs = F.avg_pool3d(hs, kernel_size=(1, 2, 2))

        ht = ht.squeeze()
        ht = torch.cat([ht, zp], 1).unsqueeze(-1).unsqueeze(-1)

        hs, hv, ht = self.block5(hs, hv, ht)
        hs, hv, ht = self.block6(hs, hv, ht)
        hv = self.fsa6(hv)
        out = hv.squeeze().transpose(1, 2)

        return out

class Z128x128_3(nn.Module):
    def __init__(self, c_a=128, c_m=10, ch=64, mode='1p2d', use_attention=True):
        super(Z128x128_3, self).__init__()

        self.use_attention = use_attention

        self.block1 = G3(c_a, c_a + c_m, c_m, ch * 8, mode, 2, 5, 1, 1, 0, 0)  #
        self.block2 = G3(ch * 8, ch * 8 * 2, ch * 8, ch * 8, mode, 1, 4, 1, 2, 0, 1)  #
        self.block3 = G3(ch * 8, ch * 8 * 2, ch * 8, ch * 4, mode, 1, 4, 1, 4, 0, 0)  #
        self.block4 = G3(ch * 4, ch * 4 * 2, ch * 4, ch * 2, mode, 1, 4, 1, 2, 0, 1)  #
        self.block5 = G3(ch * 4, ch * 4 + 3, ch * 2 + 3, ch * 2, mode, 1, 1, 1, 1, 0, 0)  #
        self.block6 = G3(ch * 2, ch * 4, ch * 2, ch, mode, 1, 1, 1, 1, 0, 0)  #

        self.fsa6 = FSA(ch * 2)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.BatchNorm3d):
                init.normal_(module.weight.data, 1.0, 0.02)
                init.constant_(module.bias.data, 0.0)

    def forward(self, za, zm, zp, query_features=None):
        zp2 = zp.argmax(dim=2)
        zp = zp.transpose(1, 2)
        # za:[1, 128, 1, 1, 1]
        # za = za.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # zm:[1, 10, 4, 1, 1]
        # zm = zm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # zv:[1, 138, 4, 1, 1]
        zv = torch.cat([za.repeat(1, 1, zm.size(2), 1, 1), zm], 1)
        # hs:[1, 512, 1, 4, 4];  hv:[1, 1024, 7, 4, 4];  ht:[1, 512, 7, 1, 1]
        hs, hv, ht = self.block1(za, zv, zm)
        # hs:[1, 512, 1, 8, 8];  hv:[1, 1024, 14, 8, 8];  ht:[1, 512, 14, 1, 1]
        hs, hv, ht = self.block2(hs, hv, ht)
        # hs:[1, 256, 1, 16, 16];  hv:[1, 512, 28, 16, 16];  ht:[1, 256, 28, 1, 1]
        hs, hv, ht = self.block3(hs, hv, ht)
        # hs:[1, 128, 1, 32, 32];  hv:[1, 256, 28, 32, 32];  ht:[1, 128, 28, 1, 1]
        hs, hv, ht = self.block4(hs, hv, ht)

        hv = F.avg_pool3d(hv, kernel_size=(1, 2, 2)).squeeze()
        hv = torch.cat([hv, zp], 1).unsqueeze(-1).unsqueeze(-1)

        # hs = F.avg_pool3d(hs, kernel_size=(1, 2, 2))
        hs = F.avg_pool3d(hs, kernel_size=(1, 2, 2)).squeeze()
        hs = torch.cat([hs, zp2], 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        ht = ht.squeeze()
        ht = torch.cat([ht, zp], 1).unsqueeze(-1).unsqueeze(-1)

        hs, hv, ht = self.block5(hs, hv, ht)
        hs, hv, ht = self.block6(hs, hv, ht)
        hv = self.fsa6(hv)
        out = hv.squeeze().transpose(1, 2)

        return out

class Z128x128_4(nn.Module):
    def __init__(self, c_a=128, c_m=10, ch=64, mode='1p2d', use_attention=True):
        super(Z128x128_4, self).__init__()

        self.use_attention = use_attention

        self.block1 = G3(c_a, c_a + c_m, c_m, ch * 8, mode, 2, 5, 1, 1, 0, 0)  #
        self.block2 = G3(ch * 8, ch * 8 * 2, ch * 8, ch * 8, mode, 1, 4, 1, 2, 0, 1)  #
        self.block3 = G3(ch * 8, ch * 8 * 2, ch * 8, ch * 4, mode, 1, 4, 1, 4, 0, 0)  #
        self.block4 = G3(ch * 4, ch * 4 * 2, ch * 4, ch * 2, mode, 1, 4, 1, 2, 0, 1)  #
        self.block5 = G3(ch * 4, ch * 4, ch * 2, ch * 2, mode, 1, 1, 1, 1, 0, 0)  #
        self.block6 = G3(ch * 2, ch * 4, ch * 2, ch, mode, 1, 1, 1, 1, 0, 0)  #

        self.fsa6 = FSA(ch * 2)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.BatchNorm3d):
                init.normal_(module.weight.data, 1.0, 0.02)
                init.constant_(module.bias.data, 0.0)

    def forward(self, za, zm, zp, query_features=None):
        zp2 = zp.argmax(dim=2)
        # za:[1, 128, 1, 1, 1]
        # za = za.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # zm:[1, 10, 4, 1, 1]
        # zm = zm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # zv:[1, 138, 4, 1, 1]
        zv = torch.cat([za.repeat(1, 1, zm.size(2), 1, 1), zm], 1)
        # hs:[1, 512, 1, 4, 4];  hv:[1, 1024, 7, 4, 4];  ht:[1, 512, 7, 1, 1]
        hs, hv, ht = self.block1(za, zv, zm)
        # hs:[1, 512, 1, 8, 8];  hv:[1, 1024, 14, 8, 8];  ht:[1, 512, 14, 1, 1]
        hs, hv, ht = self.block2(hs, hv, ht)
        # hs:[1, 256, 1, 16, 16];  hv:[1, 512, 28, 16, 16];  ht:[1, 256, 28, 1, 1]
        hs, hv, ht = self.block3(hs, hv, ht)
        # hs:[1, 128, 1, 32, 32];  hv:[1, 256, 28, 32, 32];  ht:[1, 128, 28, 1, 1]
        hs, hv, ht = self.block4(hs, hv, ht)

        hv = F.avg_pool3d(hv, kernel_size=(1, 2, 2))
        # hv = torch.cat([hv, zp], 1).unsqueeze(-1).unsqueeze(-1)

        # hs = F.avg_pool3d(hs, kernel_size=(1, 2, 2))
        hs = F.avg_pool3d(hs, kernel_size=(1, 2, 2)).squeeze()
        hs = torch.cat([hs, zp2], 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # ht = ht.squeeze()
        # ht = torch.cat([ht, zp], 1).unsqueeze(-1).unsqueeze(-1)

        hs, hv, ht = self.block5(hs, hv, ht)
        hs, hv, ht = self.block6(hs, hv, ht)
        hv = self.fsa6(hv)
        out = hv.squeeze().transpose(1, 2)

        return out


if __name__ == '__main__':
    za = torch.randn(2, 128, 1, 1, 1)
    zm = torch.randn(2, 10, 4, 1, 1)
    zp = torch.zeros(2, 128, 3)
    zp[:, 10:30, 1] = 1
    #za = za.repeat(1, 1, 3, 1, 1)
    net = Z128x128()
    out = net(za, zm, zp)
    print(out.size())
