from functools import lru_cache
import re
from os.path import abspath, dirname, join
from typing import Iterable, Optional
from warnings import warn
from .utils import load_config

import torch


class dirs:
    project_root = abspath(dirname(__file__) + "/../..")
    models = join(project_root, "models")
    plots = join(project_root, "plots")
    heatmaps = join(plots, "heatmaps")
    scatter = join(plots, "scatter")
    gifs = join(plots, "gifs")
    violins = join(plots, "violins")
    data = join(project_root, "data")
    raw_data = join(data, "raw", "aU1_ntuples")
    savepoints = join(project_root, "savepoints")
    results = join(project_root, "results")
    results_eff = join(results, "eff")
    results_latex = join(results, "latex")


class Locations:
    project_root = abspath(dirname(__file__) + "/../..")
    model = join(
        dirs.models,
        "{model}_{features}_{data_type}_{normalize}_{signal_type}_{presel_conf}"
        "_{max_norm}_{regularization}_{division}_{lepton}_{seed}.pkl",
    )
    data_two = join(dirs.data, "MC_{data_type}_{presel_conf}_two.pkl")
    data_one = join(dirs.data, "MC_{data_type}_{presel_conf}_one.pkl")
    # grid evaluation
    gridXY = join(
        dirs.savepoints,
        "gridXY_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.npz",
    )
    # plots
    train_distribution_gif = join(
        dirs.gifs,
        "training_distributions_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.gif",
    )
    heatmap = join(
        dirs.heatmaps,
        "heatmap_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.pdf",
    )
    heatmap_agg = join(
        dirs.heatmaps,
        "heatmap-agg_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.pdf",
    )
    twodim_vs_output = join(
        dirs.scatter,
        "twodim_vs_output_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.pdf",
    )
    feat_vs_output = join(
        dirs.scatter,
        "feat_vs_output_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.pdf",
    )
    roc = join(
        dirs.scatter,
        "roc_{model}_{features}_{data_type}_{normalize}_{signal_type}_{presel_conf}"
        "_{max_norm}_{regularization}_{division}_{lepton}_{seed}.pdf",
    )
    rate_vs_eff = join(
        dirs.scatter,
        "rate_vs_eff_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.pdf",
    )
    eff_vs_kinematics = join(
        dirs.scatter,
        "eff_vs_kinematics_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.pdf",
    )
    multi_eff_vs_kinematics = join(
        dirs.scatter,
        "multi_eff_vs_kinematics_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.pdf",
    )
    presel_efficiencies = join(
        dirs.results, "presel_efficiencies_{data_type}_{presel_conf}.json",
    )
    presel_efficiencies_onetrack = join(
        dirs.results, "presel_efficiencies_{data_type}_{presel_conf}_onetrack.json",
    )
    auc_acc = join(
        dirs.results,
        "auc_acc_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.json",
    )
    target_effs = join(
        dirs.results_eff,
        "target-eff_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.pkl",
    )
    target_cut = join(
        dirs.results_eff,
        "target-cut_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.txt",
    )
    full_effs = join(
        dirs.results_eff,
        "full-eff_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.pkl",
    )
    violins = join(
        dirs.violins,
        "violins_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.pdf",
    )
    eff_table = join(
        dirs.results_latex,
        "eff_table_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.txt",
    )
    exported_model = join(
        dirs.models,
        "exported_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}_{seed}.json",
    )
    seed_violins = join(
        dirs.violins,
        "seed_violins_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}_{lepton}.pdf",
    )
    onetrack_ptshift = join(
        dirs.results_eff,
        "ptshift_{data_type}_{presel_conf}_onetrack.txt",
    )
    onetrack_target_effs = join(
        dirs.results_eff,
        "target-eff_{data_type}_{presel_conf}_onetrack.pkl",
    )
    onetrack_full_effs = join(
        dirs.results_eff,
        "full-eff_{data_type}_{presel_conf}_onetrack.pkl",
    )
    onetrack_rate_vs_eff = join(
        dirs.scatter,
        "rate_vs_eff_{data_type}_{presel_conf}_onetrack.pdf",
    )


def to_string_features(features: Optional[list]) -> str:
    if features is None:
        return "None"
    return "+".join(features)


def from_string_features(features: str) -> Optional[list]:
    if features == "None":
        return None
    return features.split("+")


def to_string_normalize(normalize: Optional[bool]) -> str:
    if normalize is None:
        return "None"
    return "normed" if normalize else "unnormed"


def from_string_normalize(normalize: str) -> Optional[bool]:
    if normalize == "None":
        return None
    return normalize == "normed"


def to_string_presel_conf(presel_conf: Optional[dict]) -> str:
    if presel_conf is None:
        return "None"
    return "+".join([f"{k}:{v}" for k, v in presel_conf.items()])


def from_string_presel_conf(presel_conf: str) -> Optional[dict]:
    if presel_conf == "None":
        return None
    return {k: v for k, v in (kv.split(":") for kv in presel_conf.split("+"))}


def to_string_max_norm(max_norm: Optional[bool]) -> str:
    if max_norm is None:
        return "None"
    return "max-norm" if max_norm else "always-norm"


def from_string_max_norm(max_norm: str) -> Optional[bool]:
    if max_norm == "max-norm":
        return True
    if max_norm == "None":
        return None
    return False


def format_location(location: str, config):
    return location.format(
        model=config.model,
        features=to_string_features(config.features),
        data_type=config.data_type,
        normalize=to_string_normalize(config.normalize),
        signal_type=config.signal_type,
        presel_conf=to_string_presel_conf(config.presel_conf),
        max_norm=to_string_max_norm(config.max_norm),
        regularization=config.regularization,
        division=config.division,
        lepton=config.lepton,
        seed=config.seed,
    )


def get_cli_args(config) -> str:
    """
    config has a subset of .model, .features, .data_type, .normalize
    """
    argstr = ""
    if hasattr(config, "model"):
        argstr += f"--model={config.model} "
    if hasattr(config, "features"):
        argstr += f"--features='{from_string_features(config.features)}' "
    if hasattr(config, "data_type"):
        argstr += f"--data_type={config.data_type} "
    if hasattr(config, "normalize"):
        argstr += f"--normalize={from_string_normalize(config.normalize)} "
    if hasattr(config, "signal_type"):
        argstr += f"--signal_type={config.signal_type} "
    if hasattr(config, "presel_conf"):
        # need to double curly brace, so f strings do not work here
        argstr += (
            "--presel_conf='{"
            + str(from_string_presel_conf(config.presel_conf))
            + "}' "
        )
    if hasattr(config, "max_norm"):
        argstr += f"--max_norm={from_string_max_norm(config.max_norm)} "
    if hasattr(config, "regularization"):
        argstr += f"--regularization={config.regularization} "
    if hasattr(config, "division"):
        argstr += f"--division={config.division} "
    if hasattr(config, "seed"):
        argstr += f"--seed={config.seed} "
    if hasattr(config, "lepton"):
        argstr += f"--lepton={config.lepton} "
    return argstr


Configs = load_config(join(dirs.project_root, "config.yml"))


class Configuration:
    def __init__(
        self,
        model: Optional[str] = None,
        features: Optional[list] = None,
        normalize: Optional[bool] = None,
        data_type: Optional[str] = None,
        signal_type: Optional[str] = None,
        presel_conf: Optional[dict] = None,
        max_norm: Optional[bool] = None,
        regularization: Optional[str] = None,
        division: Optional[str] = None,
        seed: int = None,
        use_cuda: bool = Configs.use_cuda,
        sigma_final: float = Configs.sigma_final,
        sigma_init: float = Configs.sigma_init,
        plot_style: bool = Configs.plot_style,
        lepton: str = Configs.lepton,
        onetrack: bool = False,
    ):

        self.model = model
        self.features = features
        self.normalize = normalize
        self.data_type = data_type
        self.signal_type = signal_type
        self.presel_conf = presel_conf
        self.max_norm = max_norm
        self.regularization = regularization
        self.division = division
        self.seed = seed
        self.sigma_final = sigma_final
        self.sigma_init = sigma_init
        self.plot_style = plot_style
        self.lepton = lepton
        self.onetrack = onetrack

        self.device = torch.device("cpu")
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                warn("use_cuda is set to True but CUDA is unavailable...")

    def __str__(self):
        return "\n".join(
            (
                f"model={self.model}",
                f"features={self.features}",
                f"normalize={self.normalize}",
                f"data_type={self.data_type}",
                f"signal_type={self.signal_type}",
                f"presel_conf={self.presel_conf}",
                f"max_norm={self.max_norm}",
                f"regularization={self.regularization}",
                f"division={self.division}",
                f"lepton={self.lepton}",
                f"seed={self.seed}",
                f"device={self.device}",
            )
        )

    def __repr__(self):
        return self.__str__()


def expand_with_rules(location, **cfg):
    """
    snakemake like expand function,
    but conditional expansion based on the rules in valid_config.
    The location is expanded with None for a key if no rule in the
    current expansion is valid with the key configuration
    """
    for k, v in cfg.items():
        cfg[k] = list(v)
        if cfg[k] == []:
            return []

    def valid_config(cfg: dict, key: str, value):
        # rules for combinations
        # if you have a new rule to restrict combinations, add it here
        if key == "presel_conf":
            if cfg["data_type"] == "standalone":
                # standalone does not support preselections
                return False
        if key in ["max_norm", "regularization", "division"]:
            # only regularized nn models have these keywords
            regularized_models = [
                "nn-inf",
                "nn-inf-oc",
                "nn-inf-small",
                "nn-inf-mon-vchi2",
                "nn-one",
            ]
            if "model" not in cfg:
                # only consider the keywords if a regularized model is used
                if not any([m in regularized_models for m in Configs.model]):
                    return False
            elif cfg.get("model") not in regularized_models:
                return False
        if key == "features":
            if cfg.get("model") == "nn-inf-mon-vchi2":
                if len(from_string_features(value)) == 2:
                    return False
        return True

    def expand(These: Iterable[dict], key: str, With: Iterable):
        # expand configurations in a cartesian product fashion
        # with a new list
        for t in These:
            any = False
            for w in With:
                if valid_config(t, key, w):
                    any = True
                    new = t.copy()
                    new[key] = w
                    yield new
            if not any:
                yield t

    cfgs = [{}]

    for key, vals in cfg.items():
        cfgs = expand(cfgs, key, vals)

    def format_if_present(loc, **kwargs):
        # replace the existing keywords
        for k, v in kwargs.items():
            to_replace = "{" + k + "}"
            if to_replace in loc:
                loc = loc.replace(to_replace, str(v))
        # remove the ones that were not filled
        loc = re.sub("{.*?}", "None", loc)
        return loc

    out = [format_if_present(location, **cfg) for cfg in cfgs]
    return out


@lru_cache(1)
def get_config() -> Configuration:
    from fire import Fire

    return Fire(Configuration)


def get_config_from_file(file):
    obj = load_config(file)

    def default(x):
        return x[0] if type(x) == list else x

    obj_default = {k: default(v) for k, v in obj.__dict__.items()}
    return Configuration(**obj_default)


def feature_repr(feature):
    if feature == "minipchi2":
        return "log(min($\chi^2_{IP}$))"
    elif feature == "sumpt":
        return "$\sum_{tracks}p_{T}$ [GeV]"
    elif feature == "fdchi2":
        return "log($\chi^2_{FD}$)"
    elif feature == "vchi2":
        return "$\chi^2_{Vertex}$"

evttypes = list({
    11102001: "Bd_K+pi-=DecProdCut", # 2 charged basics
    11102202: "Bd_Kstgamma=HighPtGamma,DecProdCut", # 2 # not sure if i like that one
    11102405: "Bd_pi+pi-pi0=TightCuts,sqDalitz", # 2
    11104020: "Bd_phiKst0=DecProdCut", # 4
    11104041: "Bd_Kst0rho0,K+pi-pi+pi-=DecProdCut", # 4
    11110010: "Bd_Ksttaumu,3pi=DecProdCut,tauolababar,phsp", # 4
    11112001: "Bd_mumu=DecProdCut", # 2
    11114018: "Bd_ppbarmumu=DecProdCut", # 4
    11114041: "Bd_4mu=PHSP,DecProdCut", # 4
    #11114076: "Bd_KstarDarkBoson2MuMu,m=2500MeV,t=100ps,DecProdCut", # 4 # should we take those?
    #11114078: "Bd_KstarDarkBoson2MuMu,m=2500MeV,t=10ps,DecProdCut", # 4  # should we take those?
    11124001: "Bd_Kstee=DecProdCut",
    #11124002: "Bd_Kstee=btosllball05,DecProdCut",
    11144001: "Bd_JpsiKst,mm=DecProdCut",
    11144011: "Bd_psi2SKst,mm=DecProdCut",
    11144050: "Bd_JpsiKpi,mm=DecProdCut",
    #11144103: "Bd_JpsiKS,mm=CPV,DecProdCut",
    11146114: "Bd_JpsiphiKs,KKmumupipi=DecProdCut",
    11154001: "Bd_JpsiKst,ee=DecProdCut",
    11154011: "Bd_psi2SKst,ee=DecProdCut",
    11160001: "Bd_DstTauNu=DecProdCut,tauolababar",
    #11166071: "Bd_D0Kpi,Kpipipi=BsqDalitz,DAmpGen,TightCut",
    11166111: "Bd_D0Kpi,KSpipi=BsqDalitz,DDalitz,TightCut",
    11264001: "Bd_D-pi+=DecProdCut",
    11264011: "Bd_D-K+=DecProdCut",
    11274030: "Bd_Lambdacmu,pKpi=DecProdCut",
    11314001: "Bd_Kstemu=DecProdCut,PHSP",
    #11372014: "Bd_MuXMajoranaNeutrino2MuX,m=2000MeV,t=100ps,OS,DecProdCut",
    11494010: "Bd_D0D0bar,hh,hhhh=AmpGen,MINT,DecProdCut,pCut1600MeV",
    11512011: "Bd_pimunu=DecProdCut",
    11514001: "Bd_Ksttautau,mumu=DecProdCut",
    11563002: "Bd_D-taunu,Kpipi,3pinu,tauolababar=TightCut",
    11574011: "Bd_Dst+taunu,D0pi+,mununu=RDstar,TightCut",
    11574060: "Bd_D+taunu,mununu=RDplusCut",
    11574094: "Bd_Dst+munu,D0pi+=HQET2,TightCut",
    11714000: "Bd_Ksttaumu,mu=DecProdCut",
    11874092: "Bd_D0Xmunu,D0=cocktail,pipi,D0muInAcc",
    12101010: "Bu_Ktautau,3pi3pi=DecProdCut,tauola5",
    12103007: "Bu_pi+pi+pi-=sqDalitz,DecProdCut",
    12103017: "Bu_K+K+K-=sqDalitz,DecProdCut",
    12103035: "Bu_pi+K-K+=sqDalitz,DecProdCut",
    #12103101: "Bu_KSpi=DecProdCut",
    #12103121: "Bu_KsK=DecProdCut",
    12115015: "Bu_K4mu=TightCut",
    12115016: "Bu_K2mu2e=TightCut",
    12115020: "Bu_Kpipimumu=DecProdCut,LSFLAT",
    12115190: "Bu_Lambdapbarmumu=DecProdCut",
    12125000: "Bu_Kpipiee=DecProdCut,LSFLAT",
    12125190: "Bu_Lambdapbaree=DecProdCut",
    12143010: "Bu_JpsiPi,mm=DecProdCut",
    12143020: "Bu_psi2SK,mm=DecProdCut",
    #12145067: "Bu_KOmegaJpsi,mm=LSFLAT,DecProdCut",
    #12145069: "Bu_chic1K,Jpsimumu=DecProdCut",
    12153001: "Bu_JpsiK,ee=DecProdCut",
    12153020: "Bu_JpsiPi,ee=DecProdCut",
    12155111: "Bu_JpsipLambda,ee=DecProdCut",
    12163230: "Bu_Dst0pi,D0gamma,Kpi=DecProdCut",
    12163430: "Bu_Dst0pi,D0pi0,Kpi=DecProdCut",
    12165094: "Bu_Lambdacbarppi,pKpi=sqDalitz,DecProdCut",
    12165181: "Bu_D0KsPi,Kpi=DecProdCut",
    12165351: "Bu_Dst0pi,D0gamma,KSpipi=TightCut,LooserCuts",
    12167181: "Bu_D0Kst+,Kpipipi,KSpi=DecProdCut",
    12167191: "Bu_D0Kst+,KSpipi,KSpi=TightCut",
    12195032: "Bu_DstD0,D0pi+,Kpi,Kpi=DecProdCut",
    12195047: "Bu_D0Ds,Kpi,KKpi=DDalitz,DecProdCut",
    12265002: "Bu_D0PiPiPi,KPi=DecProdCut",
    12313012: "Bu_Kemu=PHSP,DecProdCut",
    #12513011: "Bu_phimunu=TightCut,BToVlnuBall",
    12513020: "Bu_3munu=DecProdCut",
    12513040: "Bu_Ktaumu,mu=DecProdCut",
    12513051: "Bu_ppmunu=DecProdCutpQCD",
    12513061: "Bu_pptaunu,mununu=DecProdCutpQCD",
    #12513070: "Bu_Higgsmumu=PPchangeDecProdCut",
    12562000: "Bu_D0taunu,Kpi,3pinu,tauolababar=DecProdCut",
    #12562410: "Bu_Dst0taunu,D0pi0,D0gamma,Kpi,3pinu,tauolababar,chargedInAcc=DecProdCut",
    12573001: "Bu_D0taunu,mununu=RDstar,TightCut",
    12715000: "Bu_Ktaumu,3pi=DecProdCut",
    #12873042: "Bu_D0Xmunu,D0=cocktail,pipi,D0muInAcc",
    12873441: "Bu_D0Xmunu,D0=cocktail",
    13102004: "Bs_K+K-=CPV2017,DecProdCut",
    #13102202: "Bs_phigamma=HighPtGamma,DecProdCut",
    13104012: "Bs_phiphi=CDFAmp,DecProdCut",
    13110007: "Bs_tautau,mu3pi=DecProdCut,tightcut,tauolababar",
    13112001: "Bs_mumu=DecProdCut",
    13114006: "Bs_phimumu=Ball,DecProdCut",
    13114020: "Bs_mumumumu=PHSP,DecProdCut",
    13114066: "Bs_ppbarmumu=DecProdCut",
    13122200: "Bs_gammaee=MNT,DecProdCut",
    13144002: "Bs_Jpsiphi,mm=CPV,DecProdCut",
    13144020: "Bs_psi2Sphi,mm=CPV,DecProdCut",
    13144031: "Bs_Jpsipipi,mm=DecProdCut",
    13144041: "Bs_JpsiKK,mm=DecProdCut",
    13154001: "Bs_Jpsiphi,ee=CPV,update2012,DecProdCut",
    13154011: "Bs_psi2Sphi,ee=CPV,DecProdCut",
    13314023: "Bs_phiemu,KK=PHSP,DecProdCut",
    13444001: "Bs_JpsiKst,mm=DecProdCut",
    13512010: "Bs_Kmunu=DecProdCut",
    #13512030: "Bs_Ktaunu,mununu=DecProdCut",
    13514030: "Bs_phitaumu,mu=Ball,DecProdCut",
    13563002: "Bs_DsTauNu,KKPi,PiPiPi=TightCut,tauolababar",
    13614041: "Bs_phitautau,mumuCocktail=TightCut",
    13774000: "Bs_Dsmunu=cocktail,hqet2,DsmuInAcc",
    13774011: "Bs_Dsmunu,KKpi,hqet2=DecProdCut",
    13774221: "Bs_Dsmunu=Ds+Dsst=hqet2,mu3hInAcc",
    15112001: "Lb_Kmu=PHSP,DecProdCut",
    15114001: "Lb_Lambda1520mumu=phsp,DecProdCut",
    15114011: "Lb_pKmumu=phsp,DecProdCut",
    15124001: "Lb_Lambda1520ee=phsp,DecProdCut",
    15124011: "Lb_pKee=phsp,DecProdCut",
    15144021: "Lb_Jpsippi,mm=phsp,DecProdCut",
    15144103: "Lb_JpsiLambda,mm=phsp,DecProdCut",
    15154001: "Lb_JpsipK,ee=phsp,DecProdCut",
    15174001: "Lb_Dsmu,KKpi=phsp,DecProdCut",
    15512014: "Lb_pmunu=DecProdCut,LQCD",
    15514042: "Lb_pKtautau,mumu=DecProdCut",
    15576011: "Lb_Lc2625munu,Lcpipi,pKpi=LHCbAcceptance",
    15863030: "Lb_Lctaunu,pKpi=cocktail,tau3pi,DecProdCut,tauolababar",
    15874041: "Lb_Lcmunu,pKpi=cocktail,Baryonlnu",
    16115139: "Omegab_Omegamumu,LambdaK=phsp,TightCut",
    16145935: "Omegab_JpsiOmega,mm,LambdaK=phsp,TightCut",
    16874040: "Xib0_Xicmunu,pKpi=cocktail",
    16875030: "Xib_Xic0munu,pKKpi=cocktail",
    16875031: "Omegab_Omegac0munu,pKKpi_PPChange=cocktail",
    21103240: "D+_etapi,pipigamma=DecProdCut",
    21263002: "D+_K-K+pi+=res,DecProdCut",
    21263010: "D+_K-pi+pi+=res,DecProdCut",
    21263020: "D+_pi-pi+pi+=res,DecProdCut",
    23103200: "Ds+_etaprimepi,rhogamma=DecProdCut",
    23103460: "Ds+_etapi,pipipi0,gg=DecProdCut",
    23123022: "Ds_phipi,ee=DecProdCut",
    23173003: "Ds_phipi,mm=FromD",
    23263020: "Ds+_K-K+pi+=res,DecProdCut",
    # 23511400: "Ds_taunu,pi0mu,gg=DecProdCut",
    # 23513016: "Ds_taunu,mmm=FromD",
    # 23513205: "Ds_taunu,etaprimemu,rhogamma=DecProdCut",
    # 23513400: "Ds_taunu,etamu,pipipi0,gg=DecProdCut",
    # 23513401: "Ds_taunu,etaprimemu,pipieta,gg=DecProdCut",
    # 23513402: "Ds_taunu,omegamu,pipipi0,gg=DecProdCut",
    25203000: "Lc_pKpi-res=LHCbAcceptance",
    26103090: "Xic+_pKpi=phsp,DecProdCut",
    26104080: "Xic0_pKKpi=phsp,DecProdCut",
    # 26104188: "Omegac0_L0KS0,ppi,pipi=pshp,DecProdCut",
    # 26104189: "Xic0_L0KS0,ppi,pipi=pshp,DecProdCut",
    27163001: "Dst_D0pi,pipi=DecProdCut",
    27163002: "Dst_D0pi,KK=DecProdCut",
    27163003: "Dst_D0pi,Kpi=DecProdCut",
    27165073: "Dst_D0pi,Kpipipi=DecProdCut,AmpGen",
    27165100: "Dst_D0pi,KSpipi=DecProdCut",
    27165101: "Dst_D0pi,KSKK=DecProdCut",
    27215002: "Dst_D0pi,KKmumu=res,DecProdCut",
    27215003: "Dst_D0pi,pipimumu=res,DecProdCut",
    27265000: "Dst_D0pi,Kpipipi=DecProdCut",
    27265101: "Dst_D0pi,KSKK=res,DecProdCut",
})



evttypes = dict((i + 1, evttype) for i, evttype in enumerate(evttypes))

input_rate = 30000 #kHz
onetrack_target_rate = 330
twotrack_target_rate = 660
