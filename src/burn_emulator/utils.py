import heapq
import importlib
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
import yaml

from pathlib import Path
from typing import Optional

from burn_emulator.constants import FBFM_OH_MAP, INPUT_KEYS, NO_DATA


def to_flow(aspect_raw: torch.Tensor, slope_deg: torch.Tensor):
    missing_mask = (aspect_raw < 0) | (slope_deg < 0)
    flat_mask = (slope_deg <= 0)
    zero_mask = missing_mask | flat_mask

    aspect_deg = aspect_raw.clamp(0, 255) * (360.0 / 256.0)
    slope_rad = torch.deg2rad(slope_deg.clamp(0.0, 47.0))
    aspect_rad = torch.deg2rad(aspect_deg)
    magnitude = torch.sin(slope_rad)
    flow_x = magnitude * torch.sin(aspect_rad)
    flow_y = -magnitude * torch.cos(aspect_rad)
    flow_x = flow_x.masked_fill(zero_mask, 0.0)
    flow_y = flow_y.masked_fill(zero_mask, 0.0)
    return flow_x, flow_y


def cache_inputs(
    fuels_paths: list[Path],
    burn_paths: list[Path],
    topo_path: list[Path],
    stats_path: Path,
    flow: bool = True
) -> tuple[dict, dict, dict]:
    inputs = {}
    topos = {}
    masks = {}
    for fuels_path, burn_path in zip(fuels_paths, burn_paths):
        fkey = fuels_path.stem
        assert burn_path.stem in fkey, f'{burn_path.stem} not in {fkey}'

        inputs[fkey] = {}
        fuels_files = sorted(list(fuels_path.glob("*.tif")))
        for file in fuels_files:
            name = file.stem.rsplit("_", 1)[1]
            if name not in INPUT_KEYS:
                continue
            with rasterio.open(file) as src:
                dat = src.read()
                if name == 'fbfm':
                    masks[fkey] = torch.tensor(dat == src.nodata)
                    masks[fkey] = torch.logical_not(masks[fkey])
                    for k, v in FBFM_OH_MAP.items():
                        dat[dat == k] = v
                    dat = F.one_hot(torch.tensor(dat).long(), num_classes=len(np.unique(list(FBFM_OH_MAP.values()))))
                    dat = dat.squeeze(0).permute(2, 0, 1)
                    dat = dat[1:]
                else:
                    dat = dat.astype(float)
                    dat[dat == src.nodata] = np.nan
                    dat[dat < 0] = 0
            inputs[fkey][name] = dat
    stats_file = stats_path.exists()

    if stats_file:
        with open(stats_path) as f:
            stats = yaml.safe_load(f)
    else:
        stats = {}
        
    for key in INPUT_KEYS:
        if key != 'fbfm':
            arrs = []
            if stats_file:
                mean = stats[key]['mean']
                stdv = stats[key]['stdv']
            else:
                for fuels_path in fuels_paths:
                    fkey = fuels_path.stem
                    arrs.append(inputs[fkey][key])
                arrs = np.concatenate(arrs)
                mean = np.nanmean(arrs).item()
                stdv = np.nanstd(arrs).item()

                stats[key] = {"mean": mean, "stdv": stdv}
            
            for fuels_path in fuels_paths:
                fkey = fuels_path.stem
                inputs[fkey][key] = (inputs[fkey][key] - mean) / stdv
                inputs[fkey][key] = torch.tensor(inputs[fkey][key])
                inputs[fkey][key][torch.isnan(inputs[fkey][key])] = NO_DATA
    
    if flow:
        with rasterio.open(topo_path / "aspect.tif") as src:
            aspect = torch.tensor(src.read()).to(float)
        with rasterio.open(topo_path / "slope_degrees.tif") as src:
            slope = torch.tensor(src.read()).to(float)
        flow_x, flow_y = to_flow(aspect, slope)
        topos['flow_x'] = flow_x
        topos['flow_y'] = flow_y
    else:
        for topo_file in topo_path.glob("*.tif"):
            tkey = topo_file.stem
            with rasterio.open(topo_file) as src:
                dat = torch.tensor(src.read())
                topos[tkey] = dat.to(float)

    if not stats_file:
        with open(stats_path, "w") as file:
            yaml.dump(stats, file, sort_keys=False)

    return inputs, topos, masks


def dynamic_import(loader: dict, kwargs: Optional[dict]=None):
    class_path = loader.get("class_path")
    init_args = loader.get("init_args", {})
    if kwargs is not None:
        init_args |= kwargs

    loader_path = class_path.rsplit(".", 1)
    module_path, class_name = loader_path
    loader_cls = getattr(importlib.import_module(module_path), class_name)
    
    return loader_cls(**init_args)
  
  
def save_checkpoint(
    model: torch.nn.Module,
    tag: str,
    epoch: int,
    step: int,
    loss: float,
    heap: list,
    outpath: Path,
) -> None:
    ckpt_dir = outpath / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_name = f"{tag}_loss-{loss:.4f}_epoch-{epoch:04d}_step-{step:06d}.pt"
    ckpt_path = ckpt_dir / ckpt_name
    
    torch.save(model.state_dict(), ckpt_path)
    heapq.heappush(heap, (-loss, epoch, step, ckpt_path.stem))

    if len(heap) > 3:
        _, _, _, worst_path = heapq.heappop(heap)
        (ckpt_dir / f"{worst_path}.pt").unlink()
