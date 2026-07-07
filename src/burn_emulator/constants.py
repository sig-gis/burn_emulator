import torch

from pathlib import Path


NO_DATA = -1
FBFM_NO_DATA = 0
DTYPE = torch.bfloat16
FBFM_OH_MAP = {-999: 0, # no data
               0: 1, # should be 91 but is in the same spot one-hot encoded anyway
               91: 1,
               101: 2,
               105: 3,
               106: 4,
               141: 5,
               144: 6,
               148: 7,
               161: 8,
               163: 9,
               186: 10,
               189: 11,
               202: 12,
               203: 13}
INF_PROFILE = {'driver': 'GTiff',
               'dtype': 'float32',
               'nodata': -999,
               'crs': "EPSG:5070",
               'blockxsize': 256,
               'blockysize': 256,
               'tiled': True,
               'compress': 'lzw',
               'interleave': 'band'
}
INPUT_KEYS = ['cbd', 'cbh', 'cc', 'fbfm', 'th']
OUTDIR = Path("data/outputs")