from lib.modules.segnext import create_segnext_t
import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, opt, pretrained=True):
        super(Model, self).__init__()
        
        self.opt = opt
        self.backbone = create_segnext_t()
        if pretrained:
            print("Pretrained is loaded ....")
            checkpoint = torch.load('./lib/pretrained/segnext_tiny_512x512_ade_160k.pth', map_location="cpu")
            self.backbone.load_state_dict(checkpoint["state_dict"])
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=150, out_channels=150, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=150),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=150, out_channels=2 * len(self.opt.categories), kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        fused_feat = self.backbone(x)
        # print(fused_feat.shape)
        out_head = self.head(fused_feat)
        P, T = out_head[:, :len(self.opt.categories), :, :], out_head[:, len(self.opt.categories):, :, :]
        B = 1 / (1 + torch.exp(-self.opt.k * (P - T)))

        if self.training:
            return P, T, B
        else:
            return P, T

if __name__ == "__main__":
    import argparse
    from configs import *
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=False, default=50, help='k')
    
    opt = parser.parse_args()
    opt.categories = categories

    model = Model(opt)
    x = torch.ones((1, 3, 700, 1000))
    P, T, B = model(x)
    print(P.shape)