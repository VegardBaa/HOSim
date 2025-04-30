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
