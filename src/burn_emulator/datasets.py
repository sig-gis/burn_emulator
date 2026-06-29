import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional

from burn_emulator.constants import NO_DATA
from burn_emulator.utils import cache_inputs


class IgnitionDataset(Dataset):
    def __init__(
        self,
        ignitions_path: Path | list[Path],
        fuels_paths: list[Path],
        burn_paths: list[Path],
        topo_path: list[Path],
        stats_path: Path,
        burn_times: int = [480],
        chip_size: int = 256,
        jitter: Optional[int] = None,
        ignition_only: bool = False,
    ) -> None:
        if Path(ignitions_path).suffix == ".csv":
            self.ignitions = pd.read_csv(ignitions_path)
        else:
            ignitions_paths = Path(ignitions_path).glob("**/*.csv")
            self.ignitions = []
            for ignitions_path in sorted(ignitions_paths):
                name = ignitions_path.stem.split("_")[0]
                ignition = pd.read_csv(ignitions_path)
                ignition = ignition.reset_index(names="ignition_number")
                ignition.loc[:, "cbp_burn"] = name
                self.ignitions.append(ignition)
            self.ignitions = pd.concat(self.ignitions)
        
        self.fuels_paths = [Path(p) for p in fuels_paths]
        self.burn_paths = [Path(p) for p in burn_paths]
        self.topo_path = Path(topo_path)
        self.stats_path = Path(stats_path)
        
        self.fuels, self.topos, self.masks = cache_inputs(self.fuels_paths, self.burn_paths, self.topo_path, self.stats_path)
        self.burn_times = [str(bt) for bt in burn_times]
        self.chip_size = chip_size
        self.jitter = jitter
        self.ignition_only = ignition_only

    def __len__(self) -> int:
        return len(self.ignitions)*len(self.burn_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sidx = idx // len(self.burn_paths)
        bidx = idx % len(self.burn_paths)
        burn_path = self.burn_paths[bidx]
        fkey = self.fuels_paths[bidx].stem

        ignition = self.ignitions.iloc[sidx]
        y = int(ignition['row'].item())
        x = int(ignition['col'].item())
        
        if self.jitter is not None:
            y += np.random.randint(-(self.jitter+1), self.jitter)
            x += np.random.randint(-(self.jitter+1), self.jitter)

        # occasionally ignitions are at a border
        S = self.chip_size // 2
        O = self.chip_size % 2
        _, H, W = self.fuels[fkey]['fbfm'].shape
        ymin, ymax = max(0, y-S), min(y+S+O, H)
        xmin, xmax = max(0, x-S), min(x+S+O, W)
        yslc = slice(ymin, ymax)
        xslc = slice(xmin, xmax)
        
        # the mask is fbfm shaped. see utils.cache_intputs
        mask = self.masks[fkey][:, yslc, xslc]
        ydiff = self.chip_size - mask.shape[1]
        xdiff = self.chip_size - mask.shape[2]
        
        if ydiff > 0:
            ypad = (0, 0, ydiff, 0) if ymin == 0 else (0, 0, 0, ydiff)
            mask = F.pad(mask, ypad, mode='constant', value=0)
        if xdiff > 0:
            xpad = (xdiff, 0, 0, 0) if xmin == 0 else (0, xdiff, 0, 0)
            mask = F.pad(mask, xpad, mode='constant', value=0)

        
        # slopes (after caching) 0: x flow 1: y flow
        arrX = []
        for key, values in self.topos.items():
            arr = values[:, yslc, xslc]
            if ydiff > 0:
                arr = F.pad(arr, ypad, mode='constant', value=NO_DATA)
            if xdiff > 0:
                arr = F.pad(arr, xpad, mode='constant', value=NO_DATA)
            arrX.append(arr)
        
        # one hots should be padded with 0 not -1
        for key, values in self.fuels[fkey].items():
            arr = values[:, yslc, xslc]
            no_data = 0 if key == "fbfm" else NO_DATA
            if ydiff > 0:
                arr = F.pad(arr, ypad, mode='constant', value=no_data)
            if xdiff > 0:
                arr = F.pad(arr, xpad, mode='constant', value=no_data)
            if key == "fbfm":
                arr_fbfm = arr
            else:
                arrX.append(arr)

        arrX = torch.concat(arrX)
        arrX = torch.concat([arrX, arr_fbfm])
        
        # burns are not necessary for inference
        if not self.ignition_only:
            arrY = []
            igd = burn_path / str(int(ignition['ignition_number'].item()))
            if self.burn_times:
                bps = [igd / bt / "fire_type.tif" for bt in self.burn_times]
            else:
                bps = [igd / "fire_type.tif"]
            for bp in bps:
                with rasterio.open(bp) as src:
                    # windowing was slower overall by a lot so reading entire array
                    # caching 10000 images was unreasonable
                    arr = torch.tensor(src.read())
                arr = arr[:, yslc, xslc]
                if ydiff > 0:
                    arr = F.pad(arr, ypad, mode='constant', value=0)
                if xdiff > 0:
                    arr = F.pad(arr, xpad, mode='constant', value=0)
                arrY.append(arr)
            arrY = torch.concat(arrY)
            arrY = (arrY >= 1)
            return arrX, arrY, mask
        else:
            return arrX, mask, (ydiff, xdiff), (ymin, ymax, xmin, xmax), (sidx, bidx)
