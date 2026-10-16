"""Models for spatiotemporal chlorophyll-a prediction and cloud gap-filling.

Two architectures share this file:

``ConvLSTM``
    A stacked peephole ConvLSTM (Shi et al., 2015). Refactored from the original
    prototype: device-safe, peephole weights registered so state_dicts load
    cleanly, optional GroupNorm/dropout, and a ``forward`` that accepts either a
    single multi-channel frame (legacy next-day mode, re-fed ``step`` times) or a
    proper day sequence ``(B, T, C, H, W)`` (gap-filling mode, iterated over T).

``PartialConvUNet``
    A partial-convolution U-Net (Liu et al., 2018) — the modern image-inpainting
    baseline. It consumes the (masked) observation stack plus a validity mask and
    reconstructs the full field, letting convolutions ignore missing pixels.

``build_model`` picks one from the config.
"""

import torch
from torch import nn
from torch.nn import functional as F


# ===========================================================================
# ConvLSTM
# ===========================================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, image_size, use_norm=False):
        super().__init__()
        assert hidden_channels % 2 == 0, "hidden_channels must be even"
        self.hidden_channels = hidden_channels
        padding = (kernel_size - 1) // 2

        def conv(in_ch, bias):
            return nn.Conv2d(in_ch, hidden_channels, kernel_size, 1, padding, bias=bias)

        self.Wxi, self.Whi = conv(input_channels, True), conv(hidden_channels, False)
        self.Wxf, self.Whf = conv(input_channels, True), conv(hidden_channels, False)
        self.Wxc, self.Whc = conv(input_channels, True), conv(hidden_channels, False)
        self.Wxo, self.Who = conv(input_channels, True), conv(hidden_channels, False)

        # peephole connections (learned, cell-state -> gate); registered here so
        # they are part of state_dict and load from checkpoints.
        h, w = image_size
        self.Wci = nn.Parameter(torch.zeros(1, hidden_channels, h, w))
        self.Wcf = nn.Parameter(torch.zeros(1, hidden_channels, h, w))
        self.Wco = nn.Parameter(torch.zeros(1, hidden_channels, h, w))

        # GroupNorm on the hidden state (new runs only; harmless for legacy load)
        self.norm = nn.GroupNorm(min(8, hidden_channels), hidden_channels) if use_norm else None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)  # input gate
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)  # forget gate
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))  # cell state
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)  # output gate
        ch = co * torch.tanh(cc)  # hidden state
        if self.norm is not None:
            ch = self.norm(ch)
        return ch, cc

    def init_hidden(self, batch_size):
        device = self.Wxi.weight.device
        _, hidden, h, w = self.Wci.shape
        zeros = torch.zeros(batch_size, hidden, h, w, device=device)
        return zeros, zeros.clone()


class ConvLSTM(nn.Module):
    """Stacked ConvLSTM with a 1x1 conv head -> single-channel field.

    forward(x):
      * x is (B, C, H, W): legacy mode — the frame is re-fed for ``step`` steps.
      * x is (B, T, C, H, W): sequence mode — iterate the recurrence over T days.
    """

    def __init__(
        self, input_channels, hidden_channels, kernel_size, image_size, step=3, use_norm=False, dropout=0.0
    ):
        super().__init__()
        self.num_layers = len(hidden_channels)
        self.step = step
        chans = [input_channels] + list(hidden_channels)
        for i in range(self.num_layers):
            setattr(
                self,
                f"cell{i}",
                ConvLSTMCell(chans[i], hidden_channels[i], kernel_size, image_size, use_norm=use_norm),
            )
        self.dropout = nn.Dropout2d(dropout) if dropout else None
        self.final_conv = nn.Conv2d(hidden_channels[-1], 1, kernel_size=1)

    def forward(self, x_in):
        if x_in.dim() == 5:  # (B, T, C, H, W) sequence
            frames = [x_in[:, t] for t in range(x_in.size(1))]
        else:  # (B, C, H, W) re-fed step times
            frames = [x_in] * self.step

        internal_state = []
        x = None
        for step, frame in enumerate(frames):
            x = frame
            for i in range(self.num_layers):
                cell = getattr(self, f"cell{i}")
                if step == 0:
                    internal_state.append(cell.init_hidden(x.size(0)))
                h, c = internal_state[i]
                x, new_c = cell(x, h, c)
                internal_state[i] = (x, new_c)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.final_conv(x)


# ===========================================================================
# Partial-convolution U-Net (image-inpainting baseline)
# ===========================================================================
class PartialConv2d(nn.Conv2d):
    """Convolution that renormalises by the fraction of valid pixels in the
    receptive field and propagates an updated validity mask (Liu et al., 2018)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("weight_mask", torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1]))
        self.slide_winsize = self.kernel_size[0] * self.kernel_size[1]

    def forward(self, x, mask):
        with torch.no_grad():
            update = F.conv2d(mask, self.weight_mask, bias=None, stride=self.stride, padding=self.padding)
            ratio = self.slide_winsize / (update + 1e-8)
            new_mask = torch.clamp(update, 0, 1)
            ratio = ratio * new_mask
        out = F.conv2d(x * mask, self.weight, self.bias, self.stride, self.padding)
        if self.bias is not None:
            b = self.bias.view(1, -1, 1, 1)
            out = (out - b) * ratio + b
        else:
            out = out * ratio
        return out * new_mask, new_mask


class _PCBlock(nn.Module):
    def __init__(self, cin, cout, down=True, use_norm=True, act="relu"):
        super().__init__()
        k, s, p = (3, 2, 1) if down else (3, 1, 1)
        self.pconv = PartialConv2d(cin, cout, k, s, p, bias=not use_norm)
        self.norm = nn.GroupNorm(min(8, cout), cout) if use_norm else None
        self.act = nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)

    def forward(self, x, m):
        x, m = self.pconv(x, m)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x), m


class PartialConvUNet(nn.Module):
    """Compact partial-convolution U-Net for masked chlorophyll inpainting.

    Input is (B, C, H, W) observations plus a (B, 1, H, W) validity mask (1 = the
    pixel is observed by at least one sensor). Output is (B, 1, H, W)."""

    def __init__(self, input_channels, base=32, depth=3):
        super().__init__()
        self.depth = depth
        chans = [input_channels] + [base * (2**i) for i in range(depth)]
        self.enc = nn.ModuleList([_PCBlock(chans[i], chans[i + 1], down=True) for i in range(depth)])
        self.dec = nn.ModuleList()
        for i in range(depth, 0, -1):
            cin = chans[i] + (chans[i - 1] if i - 1 > 0 else input_channels)
            self.dec.append(_PCBlock(cin, chans[i - 1] if i - 1 > 0 else base, down=False, act="leaky"))
        self.head = nn.Conv2d(base if depth > 0 else input_channels, 1, 1)

    def forward(self, x, mask):
        # Single shared validity mask (B,1,H,W) as in Liu et al. (2018).
        if mask.size(1) != 1:
            mask = mask.amax(dim=1, keepdim=True)
        feats = [(x, mask)]
        h, m = x, mask
        for blk in self.enc:
            h, m = blk(h, m)
            feats.append((h, m))
        # decoder with skip connections; features are concatenated, the mask is
        # unioned (max) so it stays single-channel through the partial convs.
        for i, blk in enumerate(self.dec):
            skip_x, skip_m = feats[-(i + 2)]
            h = F.interpolate(h, size=skip_x.shape[-2:], mode="nearest")
            m = F.interpolate(m, size=skip_m.shape[-2:], mode="nearest")
            h = torch.cat([h, skip_x], dim=1)
            m = torch.maximum(m, skip_m)
            h, m = blk(h, m)
        return self.head(h)


# ===========================================================================
# Factory
# ===========================================================================
def build_model(
    name, input_channels, hidden_channels, kernel_size, image_size, step=3, use_norm=False, dropout=0.0
):
    if name == "convlstm":
        return ConvLSTM(
            input_channels,
            hidden_channels,
            kernel_size,
            image_size,
            step=step,
            use_norm=use_norm,
            dropout=dropout,
        )
    if name == "pconv_unet":
        return PartialConvUNet(input_channels, base=max(16, hidden_channels[0] // 2))
    raise ValueError(f"unknown model '{name}'")
