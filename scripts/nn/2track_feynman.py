import torch
import numpy as np

import hlt2trk.data.meta_info as meta


X = torch.load(meta.locations.grid_X)
Y = torch.load(meta.locations.grid_Y)

path = f"{meta.locations.project_root}/feynman_data/"
fn = f"{meta.path_suffix}.txt"
np.savetxt(path + fn, torch.hstack((X, Y)).numpy())

from aifeynman import run_aifeynman
run_aifeynman(path, fn, 0, "14ops.txt")
