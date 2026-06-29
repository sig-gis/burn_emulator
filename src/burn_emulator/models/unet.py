import torch
import torch.nn as nn

from torch import Tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels: int, n_outputs: int, bilinear: bool = True) -> None:
        super().__init__()
        self.n_channels: int = n_channels
        self.n_outputs: int = n_outputs
        self.bilinear: bool = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor: int = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        x1: Tensor = self.inc(x)
        x2: Tensor = self.down1(x1)
        x3: Tensor = self.down2(x2)
        x4: Tensor = self.down3(x3)
        x5: Tensor = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits: Tensor = self.outc(x)
        return logits


if __name__ == '__main__':
    import time
    from torchinfo import summary
    from burn_emulator.constants import DTYPE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, dtype: {DTYPE}")

    B, GRID = 16, 128
    C_in = 19
    N_OUTPUTS = 1

    print("Building UNet")
    model = UNet(
        n_channels=C_in,
        n_outputs=N_OUTPUTS,
        bilinear=True,
    ).to(device=device)

    print("\nParameter breakdown:")
    total = sum(p.numel() for p in model.parameters())
    for name, module in [
        ('inc',   model.inc),
        ('down1', model.down1),
        ('down2', model.down2),
        ('down3', model.down3),
        ('down4', model.down4),
        ('up1',   model.up1),
        ('up2',   model.up2),
        ('up3',   model.up3),
        ('up4',   model.up4),
        ('outc',  model.outc),
    ]:
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<6} : {params:>10,}")
    print(f"  {'Total':<6} : {total:>10,}")

    model.eval()
    summary(model, input_size=(B, C_in, GRID, GRID), device=device,
            col_names=["input_size", "output_size", "num_params"],
            depth=3, mode='eval', verbose=1)

    model.to(DTYPE)
    image = torch.zeros(B, C_in, GRID, GRID, device=device, dtype=DTYPE)
    image[:, 0] = (torch.rand(B, GRID, GRID, device=device) * 2 - 1).to(DTYPE) * 0.731  # flow_x
    image[:, 1] = (torch.rand(B, GRID, GRID, device=device) * 2 - 1).to(DTYPE) * 0.731  # flow_y
    image[:, 2:] = torch.rand(B, C_in - 2, GRID, GRID, device=device).to(DTYPE)

    # warmup
    with torch.no_grad():
        _ = model(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    N_RUNS = 5
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_RUNS):
            pred = model(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    if torch.cuda.is_available():
        peak      = torch.cuda.max_memory_allocated(device) / 1024**3
        reserved  = torch.cuda.memory_reserved(device) / 1024**3
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3

        print(f"\nMemory (GiB):")
        print(f"  peak      : {peak:.2f}")
        print(f"  reserved  : {reserved:.2f}")
        print(f"  total     : {total_mem:.2f}")
        print(f"  headroom  : {total_mem - peak:.2f}")

        assert peak < 0.9 * total_mem, (
            f"Peak memory {peak:.2f} GiB exceeds 90% of device total {total_mem:.2f} GiB"
        )
    else:
        print("\nMemory check skipped (CPU)")
    

    elapsed = (t1 - t0) / N_RUNS
    print(f"\nForward pass : {elapsed*1000:.1f} ms  (mean over {N_RUNS} runs for batch size = {B})")

    print(f"Input  : ({B}, {C_in}, {GRID}, {GRID})")
    print(f"Output : {tuple(pred.shape)}")
    assert pred.shape == (B, N_OUTPUTS, GRID, GRID)

    burn_prob = torch.sigmoid(pred)
    print(f"Burn prob range : [{burn_prob.min():.3f}, {burn_prob.max():.3f}]")

    print("\nAll checks passed")