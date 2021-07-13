import numpy as np
import torch
from hlt2trk.utils.config import Locations, dirs, format_location, get_config

cfg = get_config()
fname = format_location(Locations.gridXY, cfg)
X, Y = np.load(fname).values()

path = f"{dirs.project_root}/feynman_data/"
fn = f"{fname.split('.')[0]}.txt"
np.savetxt(path + fn, np.hstack((X, Y)))

from aifeynman import run_aifeynman

run_aifeynman(path, fn, 0, "14ops.txt")
