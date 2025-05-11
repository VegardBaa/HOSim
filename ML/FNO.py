import torch
import torch.nn as nn
import torch.fft

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes        = modes     # how many low-frequency modes to keep

        # learnable complex weights: shape (in_ch, out_ch, modes)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat) * 0.1
        )

    def forward(self, x):
        # x: (B, C, N)
        B, C, N = x.shape

        # 1) FFT to get (B, C, N//2+1) complex spectrum
        x_ft = torch.fft.rfft(x, dim=-1)

        # 2) Zero-pad output spectrum and fill only low modes
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)

        # 3) Multiply low modes by our learnable weights
        #    x_ft_low: (B, C, modes), weight: (C, O, modes)
        x_ft_low = x_ft[:, :, :self.modes]
        out_ft[:, :, :self.modes] = torch.einsum("bcm,com->bom",
                                                 x_ft_low,
                                                 self.weight)

        # 4) Inverse FFT back to real space: (B, out_ch, N)
        x_out = torch.fft.irfft(out_ft, n=N, dim=-1)
        return x_out


class FNO1d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, width=64, modes=16, depth=4):
        super().__init__()
        # 1×1 conv to lift into Fourier latent dimension
        self.input_proj  = nn.Conv1d(in_channels, width, 1)

        # stacked Fourier blocks (global + local mixing)
        self.fno_blocks = nn.ModuleList()
        for _ in range(depth):
            self.fno_blocks.append(nn.ModuleList([
                SpectralConv1d(width, width, modes),
                nn.Conv1d(width, width, 1),
                nn.ReLU()
            ]))

        # back down to 1 channel
        self.output_proj = nn.Conv1d(width, out_channels, 1)

    def forward(self, x):
        # x: (B, N) → (B, 1, N)
        if x.dim() == 2:
            x = x.unsqueeze(1)  

        # lift
        x = self.input_proj(x)  # (B, width, N)

        # Fourier blocks
        for spectral, pointwise, act in self.fno_blocks:
            x = spectral(x) + pointwise(x)
            x = act(x)

        # project back
        x = self.output_proj(x)  # (B, 1, N)
        return x.squeeze(1)      # (B, N)
    
class SpectralConv2d(nn.Module):
    """
    2D Fourier layer. Keeps only the lowest modes in each dimension.
    input: (B, C, H, W)
    output: (B, O, H, W)
    """
    def __init__(self, in_channels, out_channels, modes_height, modes_width):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height
        self.modes_width = modes_width

        # Learnable complex weights: (in_ch, out_ch, modes_h, modes_w)
        self.weight_pos = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_height, modes_width, dtype=torch.cfloat) * 0.1
        )
        self.weight_neg = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_height, modes_width, dtype=torch.cfloat) * 0.1
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # 1) FFT: real to complex
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # (B, C, H, W//2+1)

        # 2) Prepare output spectrum
        H_ft, W_ft = x_ft.shape[-2], x_ft.shape[-1]
        out_ft = torch.zeros(B, self.out_channels, H_ft, W_ft,
                             dtype=torch.cfloat, device=x.device)

        # 3) Fill low-frequency modes
        #    x_ft_low: (B, C, modes_h, modes_w)
        x_ft_low = x_ft[:, :, :self.modes_height, :self.modes_width]
        #    weight: (C, O, modes_h, modes_w)
        #    output low: (B, O, modes_h, modes_w)
        out_low = torch.einsum("bchw,cohw->bohw", x_ft_low, self.weight_pos)
        out_ft[:, :, :self.modes_height, :self.modes_width] = out_low

        out_ft[:, :, -self.modes_height:, :self.modes_width] = \
        torch.einsum("bchw,cohw->bohw",
                    x_ft[:, :, -self.modes_height:, :self.modes_width],
                    self.weight_neg)

        # 4) Inverse FFT to real space
        x_out = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))
        return x_out


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator.
    Accepts inputs of shape (B, H, W) or (B, C, H, W).
    Returns outputs of shape (B, H, W) or (B, C_out, H, W).
    """
    def __init__(self, in_channels=1, out_channels=1, width=64,
                 modes_height=16, modes_width=16, depth=4):
        super().__init__()
        # 1x1 conv to lift into higher dimension
        self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)

        # Fourier blocks
        self.fno_blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.ModuleList([
                SpectralConv2d(width, width, modes_height, modes_width),
                nn.Conv2d(width, width, kernel_size=1),
                nn.ReLU()
            ])
            self.fno_blocks.append(block)

        # project back to output channels
        self.output_proj = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, H, W) or (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # to (B, 1, H, W)

        # Lift to latent dimension
        x = self.input_proj(x)  # (B, width, H, W)

        # FNO layers
        for spectral, pointwise, act in self.fno_blocks:
            x = spectral(x) + pointwise(x)
            x = act(x)

        # Project to output
        x = self.output_proj(x)  # (B, out_channels, H, W)
        # If single output channel, squeeze
        if x.shape[1] == 1:
            x = x.squeeze(1)  # (B, H, W)
        return x
# class SpectralConv2d(nn.Module):
#     """
#     2D Fourier layer. Keeps only the lowest modes in each dimension,
#     including both positive- and negative-frequency low modes.
#     input: (B, C, H, W)
#     output: (B, O, H, W)
#     """
#     def __init__(self, in_channels, out_channels, modes_height, modes_width):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes_height = modes_height
#         self.modes_width = modes_width

#         # Learnable complex weights: (in_ch, out_ch, modes_h, modes_w)
#         self.weight = nn.Parameter(
#             torch.randn(in_channels, out_channels, modes_height, modes_width, dtype=torch.cfloat) * 0.1
#         )

#     def forward(self, x):
#         # x: (B, C, H, W)
#         B, C, H, W = x.shape

#         # 1) FFT to complex spectrum: output shape (B, C, H, W//2+1)
#         x_ft = torch.fft.rfft2(x, dim=(-2, -1))

#         # 2) Prepare output spectrum
#         H_ft, W_ft = x_ft.shape[-2], x_ft.shape[-1]
#         out_ft = torch.zeros(B, self.out_channels, H_ft, W_ft,
#                              dtype=torch.cfloat, device=x.device)

#         # 3) Fill low-frequency modes in top-left (pos height, pos width)
#         x_low_pos = x_ft[:, :, :self.modes_height, :self.modes_width]
#         out_ft[:, :, :self.modes_height, :self.modes_width] = torch.einsum(
#             "bchw,cohw->bohw", x_low_pos, self.weight
#         )

#         # 4) Fill low-frequency modes in bottom-left (neg height, pos width)
#         x_low_neg = x_ft[:, :, -self.modes_height:, :self.modes_width]
#         out_ft[:, :, -self.modes_height:, :self.modes_width] = torch.einsum(
#             "bchw,cohw->bohw", x_low_neg, self.weight
#         )

#         # 5) Inverse FFT back to real space
#         x_out = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))
#         return x_out


# class FNO2d(nn.Module):
#     """
#     2D Fourier Neural Operator.
#     Accepts inputs of shape (B, H, W) or (B, C, H, W).
#     Returns outputs of shape (B, H, W) or (B, C_out, H, W).
#     """
#     def __init__(self, in_channels=1, out_channels=1, width=64,
#                  modes_height=16, modes_width=16, depth=4):
#         super().__init__()
#         # 1x1 conv to lift into latent dimension
#         self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)

#         # Fourier blocks
#         self.fno_blocks = nn.ModuleList()
#         for _ in range(depth):
#             block = nn.ModuleList([
#                 SpectralConv2d(width, width, modes_height, modes_width),
#                 nn.Conv2d(width, width, kernel_size=1),
#                 nn.ReLU()
#             ])
#             self.fno_blocks.append(block)

#         # project back to output channels
#         self.output_proj = nn.Conv2d(width, out_channels, kernel_size=1)

#     def forward(self, x):
#         # x: (B, H, W) or (B, C, H, W)
#         if x.dim() == 3:
#             x = x.unsqueeze(1)  # to (B, 1, H, W)

#         # Lift to latent dimension
#         x = self.input_proj(x)  # (B, width, H, W)

#         # FNO layers
#         for spectral, pointwise, act in self.fno_blocks:
#             x = spectral(x) + pointwise(x)
#             x = act(x)

#         # Project to output
#         x = self.output_proj(x)  # (B, out_channels, H, W)
#         if x.shape[1] == 1:
#             x = x.squeeze(1)  # (B, H, W)
#         return x


# fno/layers.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SpectralConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2):
#         super(SpectralConv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1
#         self.modes2 = modes2

#         self.scale = 1 / (in_channels * out_channels)
#         self.weights1 = nn.Parameter(
#             self.scale * torch.rand(out_channels, in_channels, self.modes1, self.modes2, dtype=torch.cfloat)
#         )
#         self.weights2 = nn.Parameter(
#             self.scale * torch.rand(out_channels, in_channels, self.modes1, self.modes2, dtype=torch.cfloat)
#         )

#     def forward(self, x):
#         """
#         Forward pass for the SpectralConv2d layer.

#         :param x: Input tensor of shape (batchsize, in_channels, height, width)
#         :return: Output tensor of shape (batchsize, out_channels, height, width)
#         """
#         batchsize, in_channels, height, width = x.shape
#         # print(f"Input x shape: {x.shape}")

#         # Perform FFT
#         x_ft = torch.fft.rfft2(x)  # [B, C, H, W_freq]
#         # print(f"x_ft shape after FFT: {x_ft.shape}")

#         # Slice the Fourier coefficients to retain only the top modes
#         x_ft_slice = x_ft[:, :, :self.modes1, :self.modes2]  # [B, C, modes1, modes2]
#         # print(f"x_ft_slice shape: {x_ft_slice.shape}")

#         # Initialize output in Fourier space
#         out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-2), x_ft.size(-1), 
#                              dtype=torch.cfloat, device=x.device)

#         # Perform complex multiplication
#         out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
#             "bixy,ioxy->boxy", x_ft_slice, self.weights1
#         )
#         out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
#             "bixy,ioxy->boxy", x_ft_slice, self.weights2
#         )

#         # Inverse FFT
#         x = torch.fft.irfft2(out_ft, s=(height, width))

#         return x

# class FNOBlock(nn.Module):
#     def __init__(self, width, modes1, modes2):
#         super(FNOBlock, self).__init__()
#         self.conv = SpectralConv2d(width, width, modes1, modes2)
#         self.w = nn.Conv2d(width, width, 1)
#         self.bn = nn.BatchNorm2d(width)
#         self.activation = nn.SiLU()

#     def forward(self, x):
#         # print("----- FNOBlock Forward Pass -----")
#         x = self.conv(x)  # [B, C, H, W]
#         x = self.bn(x)
#         x = self.activation(x)
#         x = self.w(x)
#         return x

# class FNO(nn.Module):
#     def __init__(self, in_channels, out_channels, width, modes1, modes2, layers):
#         super(FNO, self).__init__()

#         self.fc0 = nn.Linear(in_channels, width)

#         self.fno_blocks = nn.ModuleList()
#         for _ in range(layers):
#             self.fno_blocks.append(FNOBlock(width, modes1, modes2))

#         self.fc1 = nn.Linear(width, out_channels)

#     def forward(self, x):
#         if x.dim() == 3:
#             x = x.unsqueeze(1)

#         x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, width, H, W)

#         for fno in self.fno_blocks:
#             x = fno(x)

#         x = self.fc1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, out_channels, H, W)

#         return x