import torch
from torchvision.transforms import transforms
from torch import nn
from diffusers.models import AutoencoderKL


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups:int, kernel_size=3, padding=1, dropout=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(groups, out_channels, eps=1e-06, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-06, affine=True)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.nonlinearity = nn.SiLU()

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
        residual = x
        x = torch.cat([head(x) for head in self.heads], dim=1)
        x = self.conv(x)
        x = self.nonlinearity(x)
        x = x + residual
        x = self.dropout(x)
        return x

latent_dims = 4

class VariationalEncoder(nn.Module):
    def __init__(self, dropout=0., latent_dims:int=latent_dims):
        super().__init__()
        self.latent_channels = latent_dims
        self.down1 = DownsampleBlock(3, 64, 4, dropout=dropout)
        self.down2 = DownsampleBlock(64, 256, 16, dropout=dropout)
        self.down3 = DownsampleBlock(256, self.latent_channels, 1, dropout=dropout)
        
        self.mean = ResidualBlock(self.latent_channels, self.latent_channels, 1, dropout=dropout)
        self.logvar = ResidualBlock(self.latent_channels, self.latent_channels, 1, dropout=dropout)

    def forward(self, x:torch.Tensor):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar

class VariationalDecoder(nn.Module):
    def __init__(self, dropout=0., latent_dims:int=latent_dims):
        super().__init__()
        self.latent_channels = latent_dims
        self.res1 = ResidualBlock(self.latent_channels, 768, 48, dropout=dropout)
        self.up1 = UpsampleBlock(768, 768, 48, dropout=dropout)
        self.res2 = ResidualBlock(768, 512, 32, dropout=dropout)
        self.up2 = UpsampleBlock(512, 512, 32, dropout=dropout)
        self.res3 = ResidualBlock(512, 256, 16, dropout=dropout)
        self.up3 = UpsampleBlock(256, 256, 16, dropout=dropout)
        self.res4 = ResidualBlock(256, 128, 8, dropout=dropout)
        self.out = ResidualBlock(128, 3, 1, dropout=dropout)

    def forward(self, x:torch.Tensor):
        x = self.res1(x)
        x = self.up1(x)
        x = self.res2(x)
        x = self.up2(x)
        x = self.res3(x)
        x = self.up3(x)
        x = self.res4(x)
        x = self.out(x)
        return x

class StableDiffusionDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        sd_decoder = vae.decoder
        layers_to_train = ['conv_in','conv_norm_out','conv_out']
        for name, param in sd_decoder.named_parameters():
            if any([layer in name for layer in layers_to_train]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.decoder = sd_decoder
    
    def forward(self, x:torch.Tensor):
        x = self.decoder(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)

class VariationalAutoencoder(nn.Module):
    def __init__(self, use_stable_diffusion:bool, dropout=0.):
        super().__init__()
        self.encoder = VariationalEncoder(dropout)
        self.use_stable_diffusion = use_stable_diffusion
        if self.use_stable_diffusion:
            self.decoder = StableDiffusionDecoder()
        else:
            self.decoder = VariationalDecoder(dropout)
    
    def forward(self, x:torch.Tensor):
        mean, logvar = self.encoder(x)
        x = self.reparameterize(mean, logvar)
        x = self.decoder(x)
        return x, mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

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
        self.encoder = VariationalEncoder(dropout)
        self.head = DetectorHead(num_classes, self.encoder.latent_channels)
    
    def forward(self, x:torch.Tensor):
        x = self.encoder(x)
        x = self.head(x)
        return x

