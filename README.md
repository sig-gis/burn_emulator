## Overview

A quick and dirty DL training repo for burn spread.

---

### Installation

```bash
# this should be done on one of the DGX sparks we have
git clone https://github.com/sig-gis/burn_emulator.git
cd burn_emulator

# assuming you already have uv installed
uv sync
```
---

### Usage

```bash
# these should have the file paths in there
burn_emulator -m train -c path/to/model/kwargs -c path/to/training/kwargs
burn_emulator -m test -c path/to/model/kwargs -c path/to/training/kwargs

# or

sbatch slurm/train_cnns.slurm
sbatch slurm/test_cnns.slurm

```
---

### Output file path structure
```
burn_emulator/
├── data/
│   ├── configs/
│   │   └── ...
│   ├── outputs/
│   │   ├── model_name/
│   │   │   ├── checkpoints/
│   │   │   ├── ... # logs etc.
├── src/
├── ...
```
---
### Input file path structure
```
path_to_inputs/
├── ignitions_path/
│   └── ...
├── topo_path/
│   └── ... # tifs containing aspect/slope
├── fuels_path/
│   └── ... # tifs containing all inputs
├── burn_paths/
│   ├── {ignition_number}
│   │   ├── {burn_time}
│   │   │   ├── fire_type.tif
├── ...
```
