import torch
import torch.nn as nn
import torch.nn.functional as F



class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w



class ResBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.gn1   = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.gn2   = nn.GroupNorm(8, c_out)

        self.short = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.se    = SEBlock(c_out)

    def forward(self, x):
        identity = self.short(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + identity)


class UpBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ResBlock(c_in, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)



class NucleiNet(nn.Module):
    def __init__(self, in_channels=2, base=32):
        super().__init__()

        # Encoder
        self.c1 = ResBlock(in_channels, base)
        self.c2 = ResBlock(base, base*2)
        self.c3 = ResBlock(base*2, base*4)
        self.c4 = ResBlock(base*4, base*8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ResBlock(base*8, base*16),
            nn.Dropout(0.1),
            ResBlock(base*16, base*16)
        )

        # Decoder
        self.u1 = UpBlock(base*16 + base*8, base*8)
        self.u2 = UpBlock(base*8 + base*4, base*4)
        self.u3 = UpBlock(base*4 + base*2, base*2)
        self.u4 = UpBlock(base*2 + base, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):

        # Encoder
        c1 = self.c1(x)
        p1 = self.pool(c1)

        c2 = self.c2(p1)
        p2 = self.pool(c2)

        c3 = self.c3(p2)
        p3 = self.pool(c3)

        c4 = self.c4(p3)
        p4 = self.pool(c4)

        b = self.bottleneck(p4)

        u1 = self.u1(b, c4)
        u2 = self.u2(u1, c3)
        u3 = self.u3(u2, c2)
        u4 = self.u4(u3, c1)

        return self.out(u4)
    


# model = models.CellposeModel(
#     gpu=True,
#     pretrained_model=None,
#     model_type=None,   
#     nchan=2
# )

# model.net = NucleiNet(in_channels=2, base=32).to(device)