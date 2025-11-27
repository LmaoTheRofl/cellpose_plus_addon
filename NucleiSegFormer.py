import torch
from torch import nn
import segmentation_models_pytorch as smp


class NucleiSegFormer(nn.Module):

    def __init__(self,
                 encoder="mit_b5",
                 encoder_weights="imagenet",
                 decoder="MAnet",
                 diam_mean=30.0):
        super().__init__()


        Decoder = smp.MAnet if decoder == "MAnet" else smp.FPN

        self.net = Decoder(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=3,       # 1
            activation=None
        )

        self.nout = 3 # 1
        self.mkldnn = False
        self.diam_mean = nn.Parameter(torch.tensor([diam_mean]), requires_grad=False)
        self.diam_labels = nn.Parameter(torch.tensor([diam_mean]), requires_grad=False)
        
    def forward(self, X):

        if X.shape[1] < 3:
            X = torch.cat(
                (X,
                 torch.zeros((X.shape[0], 3 - X.shape[1], X.shape[2], X.shape[3]),
                             device=X.device)),
                dim=1
            )

        y = self.net(X)

        style = torch.zeros((X.shape[0], 256), device=X.device)

        return y, style


    @property
    def device(self):
        return next(self.parameters()).device
    
# model = NucleiSegFormer(
#     encoder="mit_b5",
#     encoder_weights="imagenet",
#     decoder="MAnet"
# ).cuda()