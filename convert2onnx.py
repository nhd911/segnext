import torch
from lib.modules.segnext import create_segnext_t
from dataset import *

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
    from configs import categories
    use_gpu = False
    
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.categories = categories
    opt.k = 50

    model = torch.nn.DataParallel(Model(opt))
    
    if use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    if use_gpu:
        def map_location(storage, loc): return storage.cuda()
    else:
        map_location = 'cpu'
    ckpt = torch.load("models/model_v0.pth", map_location=map_location)

    model.load_state_dict(ckpt['state_dict'])
    model.to("cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, 700, 1000).to("cpu")
    input_names = ["input"]
    output_names = ["P", "T"]

    torch.onnx.export(model.module,
                      dummy_input,
                      "models/model_v0.onnx",
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      export_params=True,
                      opset_version=11,
                      dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                    'P': {0: 'batch_size'},
                                    'T': {0: 'batch_size'}}
                      )
