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
    
# class SpectralConv2d(nn.Module):
#     """
#     2D Fourier layer. Keeps only the lowest modes in each dimension.
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

#         # 1) FFT: real to complex
#         x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # (B, C, H, W//2+1)

#         # 2) Prepare output spectrum
#         H_ft, W_ft = x_ft.shape[-2], x_ft.shape[-1]
#         out_ft = torch.zeros(B, self.out_channels, H_ft, W_ft,
#                              dtype=torch.cfloat, device=x.device)

#         # 3) Fill low-frequency modes
#         #    x_ft_low: (B, C, modes_h, modes_w)
#         x_ft_low = x_ft[:, :, :self.modes_height, :self.modes_width]
#         #    weight: (C, O, modes_h, modes_w)
#         #    output low: (B, O, modes_h, modes_w)
#         out_low = torch.einsum("bchw,cohw->bohw", x_ft_low, self.weight)
#         out_ft[:, :, :self.modes_height, :self.modes_width] = out_low

#         out_ft[:, :, -self.modes_height:, :self.modes_width] = \
#         torch.einsum("bchw,cohw->bohw",
#                     x_ft[:, :, -self.modes_height:, :self.modes_width],
#                     self.weight)

#         # 4) Inverse FFT to real space
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
#         # 1x1 conv to lift into higher dimension
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
#         # If single output channel, squeeze
#         if x.shape[1] == 1:
#             x = x.squeeze(1)  # (B, H, W)
#         return x


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer. Keeps only the lowest modes in each dimension,
    including both positive- and negative-frequency low modes.
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
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_height, modes_width, dtype=torch.cfloat) * 0.1
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # 1) FFT to complex spectrum: output shape (B, C, H, W//2+1)
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        # 2) Prepare output spectrum
        H_ft, W_ft = x_ft.shape[-2], x_ft.shape[-1]
        out_ft = torch.zeros(B, self.out_channels, H_ft, W_ft,
                             dtype=torch.cfloat, device=x.device)

        # 3) Fill low-frequency modes in top-left (pos height, pos width)
        x_low_pos = x_ft[:, :, :self.modes_height, :self.modes_width]
        out_ft[:, :, :self.modes_height, :self.modes_width] = torch.einsum(
            "bchw,cohw->bohw", x_low_pos, self.weight
        )

        # 4) Fill low-frequency modes in bottom-left (neg height, pos width)
        x_low_neg = x_ft[:, :, -self.modes_height:, :self.modes_width]
        out_ft[:, :, -self.modes_height:, :self.modes_width] = torch.einsum(
            "bchw,cohw->bohw", x_low_neg, self.weight
        )

        # 5) Inverse FFT back to real space
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
        # 1x1 conv to lift into latent dimension
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
        if x.shape[1] == 1:
            x = x.squeeze(1)  # (B, H, W)
        return x