"""U-Net model for Denoising Diffusion Probabilistic Model.
This implementation contains a number of modifications to
original U-Net (residual blocks, multi-head attention)
and also adds diffusion timestep embeddings t.
The work is based on https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement.
https://github.com/davitpapikyan/Probabilistic-Downscaling-of-Climate-Variables/
"""
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding component.
    Attributes:
        dim: Embedding dimension.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        """正弦位置编码.位置编码的是采样时刻的位置编码，而不是传统的空间位置？
        Args:
            noise_level: An array of size [B, 1] representing the difusion timesteps.
        Returns:
            Positional encodings of size [B, 1, D].
        """
        half_dim = self.dim // 2
        step = torch.arange(half_dim, dtype=noise_level.dtype, device=noise_level.device) / half_dim
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    """为什么要增加一个仿射变换？
    Transformes timestep embeddings and injects it into input tensor.
    Attributes:
        in_channels: Input tensor channels.
        out_channels: Output tensor channels.
        use_affine_level: Whether to apply an affine transformation on input or add a noise.
    """

    def __init__(self, in_channels: int, out_channels: int, use_affine_level: bool = False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Linear(in_channels, out_channels * (1+self.use_affine_level))#如果增加affine就是*2

    def forward(self, x, time_emb):
        """Forward pass.
        Args:
            x: Input tensor of size [B, D, H, W].
            time_emb: Timestep embeddings of size [B, 1, D] where D is the dimension of embedding.
        Returns:
            Transformed tensor of size [B, D, H, W].
        """
        batch_size = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(time_emb).view(batch_size, -1, 1, 1).chunk(2, dim=1)#切分成gamma beta
            # The size of gamma and beta is (batch_size, out_channels, 1, 1).
            x = (1 + gamma) * x + beta #变成ax+b的形式相当于增加了一个gama变换
        else:
            x = x + self.noise_func(time_emb).view(batch_size, -1, 1, 1)#本质上就是加一个beata项
        return x


class Upsample(nn.Module):
    """Scales the feature map by a factor of 2, i.e. upscale the feature map.
    Attributes:
        dim: Input/output tensor channels.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bicubic")
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)

    def forward(self, x):
        """Upscales the spatial dimensions of the input tensor two times.
        Args:
            x: Input tensor of size [B, 8*D, H, W].
        Returns:
            Upscaled tensor of size [B, 8*D, 2*H, 2*W].
        """
        return self.conv(self.up(x))


class Downsample(nn.Module):
    """Scale the feature map by a factor of 1/2, i.e. downscale the feature map.
    Attributes:
        dim: Input/output tensor channels.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """Downscales the spatial dimensions of the input tensor two times.
        Args:
            x: Input tensor of size [B, D, H, W].
        Returns:
            Downscaled tensor of size [B, D, H/2, W/2].
        """
        return self.conv(x)


class Block(nn.Module):
    """残差blockA building component of Residual block.
    Attributes:
        dim: Input tensor channels.
        dim_out: Output tensor channels.
        groups: Number of groups to separate the channels into.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, dim_out: int, groups: int = 32, dropout: float = 0):
        super().__init__()
        self.block = nn.Sequential(nn.GroupNorm(num_groups=groups, num_channels=dim),
                                   nn.SiLU(),
                                   nn.Dropout2d(dropout) if dropout != 0 else nn.Identity(),
                                   nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1))

    def forward(self, x):
        """Applies block transformations on input tensor.
        Args:
            x: Input tensor of size [B, D, H, W].
        Returns:
            Transformed tensor of size [B, D, H, W].
        """
        return self.block(x)


class ResnetBlock(nn.Module):
    """Residual block.
    Attributes:
        dim: Input tensor channels.
        dim_out: Output tensor channels.
        noise_level_emb_dim: Timestep embedding dimension.
        dropout: Dropout probability.
        use_affine_level: Whether to apply an affine transformation on input or add a noise.
        norm_groups: The number of groups for group normalization.
    """

    def __init__(self, dim: int, dim_out: int, noise_level_emb_dim: int = None, dropout: float = 0,
                 use_affine_level: bool = False, norm_groups: int = 32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(in_channels=noise_level_emb_dim, out_channels=dim_out,
                                            use_affine_level=use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=1) \
            if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        """Applied Residual block on input tensors.
        Args:
            x: Input tensor of size [B, D, H, W].
            time_emb: Timestep embeddings of size [B, 1, D] where D is the dimension of embedding.
        Returns:
            Transformed tensor of size [B, D, H, W].
        """
        h = self.block1(x)  # 编码
        h = self.noise_func(h, time_emb) #仿射变换
        h = self.block2(h) #编码
        return h + self.res_conv(x) #编码+残差


class SelfAttention(nn.Module):
    """Multi-head attention.
    Attributes:
        in_channel: Input tensor channels.
        n_head: The number of heads in multi-head attention.
        norm_groups: The number of groups for group normalization.
    """

    def __init__(self, in_channel: int, n_head: int = 1, norm_groups: int = 32):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channels=in_channel, out_channels=3*in_channel, kernel_size=1, bias=False)
        self.out = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)

    def forward(self, x):
        """Applies self-attention to input tensor.
        Args:
            x: Input tensor of size [B, 8*D, H, W].
        Returns:
            Transformed tensor of size [B, 8*D, H, W].
        """
        batch_size, channel, height, width = x.shape
        head_dim = channel // self.n_head

        norm = self.norm(x)
        qkv = self.qkv(norm).view(batch_size, self.n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch_size, self.n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch_size, self.n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()#多头后用爱因斯坦和加一起
        out = self.out(out.view(batch_size, channel, height, width))

        return out + x #多头后残差链接


class ResnetBlocWithAttn(nn.Module):
    """ResnetBlock combined with sefl-attention layer.
    Attributes:
        dim: Input tensor channels.
        dim_out: Output tensor channels.
        noise_level_emb_dim: Timestep embedding dimension.
        norm_groups: The number of groups for group normalization.
        dropout: Dropout probability.
        with_attn: Whether to add self-attention layer or not.
    """

    def __init__(self, dim: int, dim_out: int, *, noise_level_emb_dim: int = None,
                 norm_groups: int = 32, dropout: float = 0, with_attn: bool = True):
        super().__init__()
        self.res_block = ResnetBlock(dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        self.attn = SelfAttention(dim_out, norm_groups=norm_groups) if with_attn else nn.Identity()

    def forward(self, x, time_emb):
        """Forward pass.
        Args:
            x: Input tensor of size [B, D, H, W].
            time_emb: Timestep embeddings of size [B, 1, D] where D is the dimension of embedding.
        Returns:
            Transformed tensor of size [B, D, H, W].
        """
        x = self.res_block(x, time_emb)
        x = self.attn(x)
        return x


class UNet(nn.Module):
    """Defines U-Net network.
    Attributes:
        in_channel: Input tensor channels.
        out_channel: Output tensor channels.
        inner_channel: Timestep embedding dimension.
        norm_groups: The number of groups for group normalization.
        channel_mults: A tuple specifying the scaling factors of channels.
        attn_res: A tuple of spatial dimensions indicating in which resolutions to use self-attention layer.
        res_blocks: The number of residual blocks.
        dropout: Dropout probability.
        with_noise_level_emb: Whether to apply timestep encodings or not.
        height: Height of input tensor.
    """

    def __init__(self, in_channel: int, out_channel: int, inner_channel: int,
                 norm_groups: int, channel_mults: tuple, attn_res: tuple,
                 res_blocks: int, dropout: float, with_noise_level_emb: bool = True, height: int = 128):
        super().__init__()
        self.out_channel=out_channel
        if with_noise_level_emb:
            noise_level_channel = inner_channel

            # Time embedding layer that returns
            self.time_embedding = nn.Sequential(PositionalEncoding(inner_channel),
                                                nn.Linear(inner_channel, 4*inner_channel),
                                                nn.SiLU(),
                                                nn.Linear(4*inner_channel, inner_channel))
        else:
            noise_level_channel, self.time_embedding = None, None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        current_height = height
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]

        for ind in range(num_mults):  # For each channel growing factor.
            is_last = (ind == num_mults - 1)

            use_attn = current_height in attn_res
            channel_mult = inner_channel * channel_mults[ind]

            for _ in range(res_blocks):  # Add res_blocks number of ResnetBlocWithAttn layer.
                downs.append(ResnetBlocWithAttn(pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                                                norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult

            # If the newly added ResnetBlocWithAttn layer to downs list is not the last one,
            # then add a Downsampling layer.
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                current_height //= 2

        self.downs = nn.ModuleList(downs)
        self.mid = nn.ModuleList([ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                                                     norm_groups=norm_groups, dropout=dropout),
                                  ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                                                     norm_groups=norm_groups, dropout=dropout, with_attn=False)])

        ups = []
        for ind in reversed(range(num_mults)):  # For each channel growing factor (in decreasing order).
            is_last = (ind < 1)
            use_attn = (current_height in attn_res)
            channel_mult = inner_channel * channel_mults[ind]

            for _ in range(res_blocks+1):  # Add res_blocks+1 number of ResnetBlocWithAttn layer.
                ups.append(ResnetBlocWithAttn(pre_channel+feat_channels.pop(), channel_mult,
                                              noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                              dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult

            # If the newly added ResnetBlocWithAttn layer to ups list is not the last one,
            # then add an Upsample layer.
            if not is_last:
                ups.append(Upsample(pre_channel))
                current_height *= 2

        self.ups = nn.ModuleList(ups)

        # Final convolution layer to transform the spatial dimensions to the desired shapes.
        self.final_conv = Block(pre_channel, out_channel if out_channel else in_channel, groups=norm_groups)

    def forward(self, x, time):
        """Forward pass.
        Args:
            x: Input tensor of size: [B, C, H, W], for WeatherBench C=2.
            time: Diffusion timesteps of size: [B, 1].
        Returns:
            Estimation of Gaussian noise.
        """
        
        t = self.time_embedding(time) if self.time_embedding else None  # [B, 1, D]
        feats = []

        for layer in self.downs:
            x = layer(x, t) if isinstance(layer, ResnetBlocWithAttn) else layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, t) if isinstance(layer, ResnetBlocWithAttn) else layer(x)

        # for i in feats:
        #     print("feats",i.shape)
        for layer in self.ups:
            # print(x.shape)
            if  x.shape[3] == 176 :
                x=x[:,:,:,:-1]
            elif x.shape[3]==36:
                x=x[:,:,:,:-1]
            x = layer(torch.cat((x, feats.pop()), dim=1), t) if isinstance(layer, ResnetBlocWithAttn) else layer(x)

        return self.final_conv(x)

if __name__ == '__main__':
    import numpy as np 
    model=UNet(10,5,64,32,[1,2,4,8],attn_res=[16],res_blocks=1,dropout=0.6)
    print(model)
    # print(model(torch.randn(2,10,300,400),torch.arange(1000)).shape)