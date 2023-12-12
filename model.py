import torch
from torchvision.transforms import transforms
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups:int, kernel_size=3, padding=1, dropout=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(groups, out_channels, eps=1e-06, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-06, affine=True)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        residual = x
        x = self.nonlinearity(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = x + residual
        return x

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups:int, kernel_size=3, padding=1, dropout=0.):
        super().__init__()
        self.resblock = ResidualBlock(in_channels, out_channels, groups, kernel_size, padding, dropout)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.nonlinearity = nn.SiLU()

    def forward(self, x):
        x = self.resblock(x)
        x = self.downsample(x)
        x = self.nonlinearity(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups, kernel_size=3, padding=1, dropout=0.):
        super().__init__()
        self.resblock = ResidualBlock(in_channels, out_channels, groups, kernel_size, padding, dropout)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.nonlinearity = nn.SiLU()

    def forward(self, x):
        x = self.resblock(x)
        x = self.upsample(x)
        x = self.nonlinearity(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, in_channels, dropout=0.):
        super().__init__()
        self.k = nn.Linear(in_channels, in_channels)
        self.q = nn.Linear(in_channels, in_channels)
        self.v = nn.Linear(in_channels, in_channels)
        self.nonlinearity = nn.SiLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).transpose(1,2)
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        attention = torch.bmm(q, k.transpose(1,2)) / (C**0.5)
        attention = self.softmax(attention)
        x = torch.bmm(attention, v).transpose(1,2).reshape(B, C, H, W)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, in_channels, num_heads, dropout=0.):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(in_channels, dropout) for _ in range(num_heads)])
        self.conv = ResidualBlock(in_channels*num_heads, in_channels, 1)
        self.nonlinearity = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=1)
        x = self.conv(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        return x

latent_dims = 4

class Encoder(nn.Module):
    def __init__(self, dropout=0., latent_dims:int=latent_dims):
        super().__init__()
        self.latent_channels = latent_dims
        self.down1 = DownsampleBlock(3, 64, 4, dropout=dropout)
        self.down2 = DownsampleBlock(64, 128, 8, dropout=dropout)
        #self.down3 = DownsampleBlock(128, 256, 16, dropout=dropout)
        self.down3 = DownsampleBlock(128, self.latent_channels, 1, dropout=dropout)
        #self.attn = MultiheadAttention(self.latent_channels, 32, dropout=dropout)

    def forward(self, x:torch.Tensor):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        # x = self.down4(x)
        # xa = self.attn(x)
        # x = xa + x
        return x

class Decoder(nn.Module):
    def __init__(self, dropout=0., latent_dims:int=latent_dims):
        super().__init__()
        self.latent_channels = latent_dims
        #self.up1 = UpsampleBlock(self.latent_channels, 768, 48, dropout=dropout)
        self.res1 = ResidualBlock(self.latent_channels, 768, 48, dropout=dropout)
        self.up2 = UpsampleBlock(768, 512, 32, dropout=dropout)
        self.res2 = ResidualBlock(512, 512, 32, dropout=dropout)
        self.up3 = UpsampleBlock(512, 256, 16, dropout=dropout)
        self.res3 = ResidualBlock(256, 256, 16, dropout=dropout)
        self.up4 = UpsampleBlock(256, 128, 8, dropout=dropout)
        self.res4 = ResidualBlock(128, 64, 4, dropout=dropout)
        self.out = ResidualBlock(64, 3, 1, dropout=dropout)

    def forward(self, x:torch.Tensor):
        # x = self.up1(x)
        x = self.res1(x)
        x = self.up2(x)
        x = self.res2(x)
        x = self.up3(x)
        x = self.res3(x)
        x = self.up4(x)
        x = self.res4(x)
        x = self.out(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.encoder = Encoder(dropout)
        self.decoder = Decoder(dropout)
        self.activation = nn.SiLU()#nn.Hardtanh(0,1)
    
    def forward(self, x:torch.Tensor):
        x = self.encoder(x)
        encoded = x
        x = self.decoder(x)
        x = self.activation(x)
        return x, encoded

class LatentResizer(nn.Module):
    def __init__(self, target_size = (3,3)) -> None:
        super().__init__()
        self.latent_dims = target_size
        self.transform = transforms.Resize(target_size)
    
    def forward(self, x:torch.Tensor):
        return self.transform(x)

class DetectorHead(nn.Module):
    def __init__(self, num_classes:int, latent_channels:int) -> None:
        # predicts class, objectness, and one bounding box
        super().__init__()
        self.resize_latent = LatentResizer()
        self.input_dims = self.resize_latent.latent_dims
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.input_dims[0]*self.input_dims[1]*latent_channels, num_classes + 5)
    
    def forward(self, x:torch.Tensor):
        x = self.resize_latent(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

class Detector(nn.Module):
    def __init__(self, num_classes:int, dropout=0.) -> None:
        super().__init__()
        self.encoder = Encoder(dropout)
        self.head = DetectorHead(num_classes, self.encoder.latent_channels)
    
    def forward(self, x:torch.Tensor):
        x = self.encoder(x)
        x = self.head(x)
        return x

