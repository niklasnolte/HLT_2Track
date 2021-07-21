from hlt2trk.utils.data import get_data_for_training
from hlt2trk.utils.config import get_config
from train_torch_model import train_torch_model
from train_bdt_model import train_bdt_model
from train_simple_model import train_simple_model

cfg = get_config()

x_train, y_train, x_val, y_val = get_data_for_training(cfg)

if cfg.model in ["regular", "sigma", "sigma-safe"]:
    train_torch_model(cfg, x_train, y_train, x_val, y_val)
if cfg.model == "bdt":
    train_bdt_model(cfg, x_train, y_train, x_val, y_val)
if cfg.model in ["lda", "qda", "gnb"]:
    train_simple_model(cfg, x_train, y_train, x_val, y_val)
