import numpy as np
import pandas as pd
import rasterio
import time
import torch
import torch.nn.functional as F

from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader

from burn_emulator.constants import DTYPE, INF_PROFILE, OUTDIR
from burn_emulator.utils import dynamic_import


def _drain(pending: list[Future], limit: int) -> None:
    while len(pending) >= limit:
        pending.pop(0).result()


def _unpack_sample(b: int, diffs: tuple, slices: tuple) -> tuple[int, int, int, int, int, int]:
    ydiff, xdiff = diffs
    ymin, ymax, xmin, xmax = slices
    yd, xd = int(ydiff[b]), int(xdiff[b])
    y0, y1 = int(ymin[b]), int(ymax[b])
    x0, x1 = int(xmin[b]), int(xmax[b])
    h, w = y1 - y0, x1 - x0
    y_start = yd if (yd > 0 and y0 == 0) else 0
    x_start = xd if (xd > 0 and x0 == 0) else 0
    return y0, y1, x0, x1, y_start, x_start, h, w


def _accumulate_batch(
    pred: torch.Tensor,
    diffs: tuple,
    slices: tuple,
    n_count: torch.Tensor,
    mean_acc: torch.Tensor,
    M2_acc: torch.Tensor,
    log_prod_acc: torch.Tensor,
    log1m_prod_acc: torch.Tensor,
) -> None:
    for b in range(pred.shape[0]):
        y0, y1, x0, x1, y_start, x_start, h, w = _unpack_sample(b, diffs, slices)

        x = pred[b, :, y_start:y_start + h, x_start:x_start + w].double()
        roi = (slice(None), slice(y0, y1), slice(x0, x1))

        n_count[roi] += 1
        delta = x - mean_acc[roi]
        mean_acc[roi] += delta / n_count[roi]
        delta2 = x - mean_acc[roi]
        M2_acc[roi] += delta * delta2

        eps = 1e-7
        log_prod_acc[roi] += torch.log(x.clamp(min=eps))
        log1m_prod_acc[roi] += torch.log((1.0 - x).clamp(min=eps))


def _write_monte_carlo(
    n_count: torch.Tensor,
    mean_acc: torch.Tensor,
    M2_acc: torch.Tensor,
    log_prod_acc: torch.Tensor,
    log1m_prod_acc: torch.Tensor,
    mask: torch.Tensor,
    test_name: str,
    outpath: Path,
    profile: dict,
) -> None:
    mean_out = mean_acc.float() * mask
    std_out = (M2_acc / n_count.clamp(min=1)).sqrt().float() * mask
    entropy_out = (
        -(log_prod_acc / n_count.clamp(min=1))
        - (log1m_prod_acc / n_count.clamp(min=1))
    ).float() * mask
    n_out = n_count.float() * mask
    with rasterio.open(outpath / f'{test_name}_entropy.tif', 'w', **profile) as dst:
        dst.write(entropy_out.cpu().numpy())
    with rasterio.open(outpath / f'{test_name}_mean.tif', 'w', **profile) as dst:
        dst.write(mean_out.cpu().numpy())
    with rasterio.open(outpath / f'{test_name}_stdv.tif', 'w', **profile) as dst:
        dst.write(std_out.cpu().numpy())
    with rasterio.open(outpath / f'{test_name}_count.tif', 'w', **profile) as dst:
        dst.write(n_out.cpu().numpy())


def _write_batch(
    pred: np.ndarray,
    diffs: tuple,
    slices: tuple,
    idxs: tuple,
    ignitions: Any,
    shape: tuple,
    profile: dict,
    test_name: str,
    outpath: Path,
) -> None:
    sidx, _ = idxs

    for b in range(pred.shape[0]):
        y0, y1, x0, x1, y_start, x_start, h, w = _unpack_sample(b, diffs, slices)

        canvas = torch.zeros(shape, dtype=torch.float32)
        canvas[:, y0:y1, x0:x1] = torch.from_numpy(pred[b, :, y_start:y_start + h, x_start:x_start + w])

        ignition_number = str(ignitions.iloc[int(sidx[b])]['ignition_number'])
        cbp_burn = str(ignitions.iloc[int(sidx[b])]['cbp_burn'])
        sample_path = outpath / cbp_burn / ignition_number / f'{test_name}.tif'
        sample_path.parent.mkdir(exist_ok=True, parents=True)

        with rasterio.open(sample_path, 'w', **profile) as dst:
            dst.write(canvas.numpy())


def test_model(
    model: torch.nn.Module,
    num_sims: int,
    test_loader: DataLoader,
    test_name: str,
    monte_carlo: bool,
    outpath: Path,
    max_write_workers: int = 4,
) -> None:
    model.to('cuda', dtype=DTYPE)
    model.eval()

    # how fragile things can be...
    fuels_path0 = list(test_loader.dataset.fuels_paths[0].glob("*fbfm*.tif"))[0]
    with rasterio.open(fuels_path0) as src:
        height, width = src.height, src.width
        transform = src.transform
    bts = test_loader.dataset.burn_times
    count = len(bts) if bts else 1
    shape = (count, height, width)
    profile = INF_PROFILE.copy()
    profile.update({"height": height,
                    "width": width,
                    "count": count,
                    "transform": transform})

    if monte_carlo:
        odtype = torch.float64
        n_count = torch.zeros([*shape], dtype=odtype, device='cuda')
        mean_acc = torch.zeros([*shape], dtype=odtype, device='cuda')
        M2_acc = torch.zeros([*shape], dtype=odtype, device='cuda')
        log_prod_acc = torch.zeros([*shape], dtype=odtype, device='cuda')
        log1m_prod_acc = torch.zeros([*shape], dtype=odtype, device='cuda')
    else:
        odtype = torch.float32

    pending: list[Future] = []
    sim_perf_times = []
    sam_perf_times = [] # does not account for partial batches
    drn_perf_times = []
    
    test_start_time = time.perf_counter()
    with torch.no_grad(), ThreadPoolExecutor(max_workers=max_write_workers) as pool:
        for _ in range(num_sims):
            sim_start_time = time.perf_counter()
            for X, M, diffs, slices, idxs in test_loader:
                X = X.to('cuda', dtype=DTYPE)
                M = M.to('cuda', dtype=DTYPE)
                
                sam_start_time = time.perf_counter()
                pred = (F.sigmoid(model(X)) * M).to(odtype)
                sam_end_time = time.perf_counter()
                sam_perf_times.append(sam_end_time-sam_start_time)
                
                if monte_carlo:
                    _accumulate_batch(
                        pred, diffs, slices,
                        n_count, mean_acc, M2_acc,
                        log_prod_acc, log1m_prod_acc,
                    )
                else:
                    drn_start_time = time.perf_counter()
                    _drain(pending, limit=max_write_workers)
                    drn_end_time = time.perf_counter()
                    drn_perf_times.append(drn_end_time-drn_start_time)
                    
                    pending.append(pool.submit(
                        _write_batch,
                        pred.cpu().numpy(),
                        diffs, slices, idxs,
                        test_loader.dataset.ignitions,
                        shape, profile.copy(),
                        test_name, outpath,
                    ))
            sim_end_time = time.perf_counter()
            sim_perf_times.append(sim_end_time-sim_start_time)
        _drain(pending, limit=1)
    if monte_carlo:
        mask = test_loader.dataset.masks[fuels_path0.stem].to('cuda')
        _write_monte_carlo(
            n_count, mean_acc, M2_acc,
            log_prod_acc, log1m_prod_acc,
            mask, test_name, outpath, profile,
        )
    test_end_time = time.perf_counter()
    test_perf_time = test_end_time-test_start_time
    tp = {"model": test_name,
          "num_batches": len(test_loader),
          "batch_size": test_loader.batch_size,
          "max_memory_alloc": np.round(torch.cuda.max_memory_allocated('cuda') / 1024**3, decimals=2),
          "test_perf_time": np.round(test_perf_time, decimals=2),
          "sim_perf_time_mu": np.round(np.mean(sim_perf_times), decimals=2).item(),
          "sam_perf_time_mu": np.round(np.mean(sam_perf_times), decimals=2).item(),
          "drn_perf_time_mu": np.round(np.mean(drn_perf_times), decimals=2).item()}
    df = pd.DataFrame([tp])
    header = False if (outpath / 'throughput.csv').exists() else True
    df.to_csv(outpath / 'throughput.csv', mode='a', index=False, header=header)


def test(**kwargs: Any) -> None:
    model = dynamic_import(kwargs.get("model"))
    dataset = dynamic_import(kwargs.get("dataset"))
    test_loader = dynamic_import(kwargs.get("dataloader"), {"dataset": dataset})

    ckpt_dir = OUTDIR / kwargs.get("model_name") / "checkpoints"
    ckpt_path = sorted(ckpt_dir.glob("*.pt"))[0]
    ckpt = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(ckpt)

    test_name = f'{kwargs.get("model_name")}_{kwargs.get("test_name")}'
    test_model(
        model=model,
        num_sims=kwargs.get("num_sims"),
        test_loader=test_loader,
        test_name=test_name,
        monte_carlo=kwargs.get("monte_carlo"),
        outpath=OUTDIR / "inference",
    )