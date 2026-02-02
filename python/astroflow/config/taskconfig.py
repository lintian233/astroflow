import os
from sympy import preorder_traversal
import yaml
import urllib.request
import hashlib


from .rficonfig import RFIConfig, IQRMConfig

CENTERNET = 0
YOLOV11N = 1
DETECTNET = 2
COMBINENET = 3


class TaskConfig:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TaskConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file=None):
        if self._initialized:
            return
        self._initialized = True
        self._rficonfig = None
        self._iqrmcfg = None
        self.config_file = config_file
        self._config_data = self._load_config()
        cputhread = self._config_data.get("cputhread", 16)
        os.environ['OMP_NUM_THREADS'] = f'{cputhread}'
        
    def _load_config(self):
        # 定义默认配置，直接写死在代码里
        default_config = {
            "cputhread": 16,
            "modelname": "yolov11n",
            "mode": "single",
            "batchsize": 128,
            "plotworker": 1,
            "dmtconfig": {"minpercentile": 0, "maxpercentile": 100},
            "specconfig": {"minpercentile": 0, "maxpercentile": 100, "onlyspec": False},
            "tsample": [
                {"name": "default", "t": 0.5}
            ],
            "confidence": 0.5,
            "dedgpu": 0,
            "detgpu": 0,
            "rfi": {
                "use_mask": False,
                "use_zero_dm": False,
                "use_iqrm": True,
            },
            "iqrm": {
                "mode": 1,
                "radius_frac": 0.1,
                "nsigma": 10.0,
                "geofactor": 1.5,
                "win_sec": 0,
                "hop_sec": 0.5,
                "include_tail": True,
            }
        }

        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, "r") as file:
                return yaml.safe_load(file)
        else:
            print("⚠️ Warning: NO config file found; Use Default Config, Maybe core dump ⚠️")
            return default_config


    def _checker_tsample(self, tsample):
        if isinstance(tsample, list):
            for item in tsample:
                if not isinstance(item, dict) or not all(
                    key in item for key in ["name", "t"]
                ):
                    raise ValueError("Invalid format for tsample in config file.")
                if not isinstance(item["name"], str):
                    raise ValueError("Name in tsample must be a string.")
                if not isinstance(item["t"], (int, float)):
                    raise ValueError("t in tsample must be a number.")
        else:
            raise ValueError("Invalid format for tsample in config file.")

    def _checker_dmrange(self, dmrange):
        if isinstance(dmrange, list):
            for item in dmrange:
                if not isinstance(item, dict) or not all(
                    key in item for key in ["name", "dm_low", "dm_high", "dm_step"]
                ):
                    raise ValueError("Invalid format for dmrange in config file.")
                if not isinstance(item["name"], str):
                    raise ValueError("Name must be a string.")
                if not all(
                    isinstance(item[key], (int, float))
                    for key in ["dm_low", "dm_high", "dm_step"]
                ):
                    raise ValueError("dm_low, dm_high, and dm_step must be numbers.")
        else:
            raise ValueError("Invalid format for dmrange in config file.")

    def _checker_freqrange(self, freqrange):
        if isinstance(freqrange, list):
            for item in freqrange:
                if not isinstance(item, dict) or not all(
                    key in item for key in ["name", "freq_start", "freq_end"]
                ):
                    raise ValueError("Invalid format for freqrange in config file.")
                if not isinstance(item["name"], str):
                    raise ValueError("Name in freqrange must be a string.")
                if not all(
                    isinstance(item[key], (int, float))
                    for key in ["freq_start", "freq_end"]
                ):
                    raise ValueError(
                        "freq_start and freq_end in freqrange must be numbers."
                    )
        else:
            raise ValueError("Invalid format for freqrange in config file.")

    # def _checker_preprocess(self, preprocess):
    #     if isinstance(preprocess, list):
    #         for item in preprocess:
    #             if not isinstance(item, dict) or len(item) != 1:
    #                 raise ValueError(
    #                     "Invalid format for preprocess in config file. Each item should be a single key-value pair."
    #                 )
    #     else:
    #         raise ValueError("Invalid format for preprocess in config file.")
    @property
    def gencand(self):
        gencand = self._config_data.get("gencand")
        if gencand is None:
            gencand = False
        if not isinstance(gencand, bool):
            raise ValueError("gencand must be a boolean.")
        return gencand
    @property
    def minfileage(self):
        minfileage = self._config_data.get("minfileage")
        if minfileage is None:
            minfileage = 0
        if not isinstance(minfileage, (int, float)):
            raise ValueError("minfileage must be a number.")
        if minfileage < 0:
            raise ValueError("minfileage must be non-negative.")
        return minfileage

    @property
    def onlycand(self):
        onlycand = self._config_data.get("onlycand")
        if onlycand is None:
            onlycand = False
        if not isinstance(onlycand, bool):
            raise ValueError("onlycand must be a boolean.")
        return onlycand

    def get_model(self):
        model_url_path = "https://github.com/lintian233/astroflow/releases/download/v0.1.1/yolo11n_0816_v1.pt"
        expected_sha256 = "d4305e273fec6f5733f3c0a823fa5275064b015d549fda26529e0b1b8f59c124" # SHA256 for yolo11n_0816_v1.pt
        max_retries = 3
        
        config_dir = os.path.expanduser("~/.config/astroflow")
        os.makedirs(config_dir, exist_ok=True)
        
        model_filename = os.path.basename(model_url_path)
        local_model_path = os.path.join(config_dir, model_filename)
        
        for attempt in range(max_retries):
            # --- Download if needed ---
            if not os.path.exists(local_model_path):
                print(f"Downloading model from {model_url_path}...")
                try:
                    urllib.request.urlretrieve(model_url_path, local_model_path)
                    print(f"Model downloaded to {local_model_path}")
                except Exception as e:
                    print(f"Warning: Failed to download model on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt + 1 == max_retries:
                        raise IOError(f"Failed to download model from {model_url_path} after {max_retries} attempts.")
                    continue # Retry

            # --- Validation ---
            if not os.path.exists(local_model_path) or os.path.getsize(local_model_path) == 0:
                print(f"Warning: Model file is missing or empty on attempt {attempt + 1}/{max_retries}. Retrying download.")
                if os.path.exists(local_model_path):
                    os.remove(local_model_path)
                continue # Retry

            with open(local_model_path, "rb") as f:
                sha256_hash = hashlib.sha256(f.read()).hexdigest()
            
            if sha256_hash == expected_sha256:
                print("Model checksum verified successfully.")
                return local_model_path # Success
            else:
                print(
                    f"Warning: Model file checksum mismatch on attempt {attempt + 1}/{max_retries}. "
                    f"Expected {expected_sha256}, but got {sha256_hash}. "
                    "The file is corrupted and will be re-downloaded."
                )
                os.remove(local_model_path) # Remove corrupted file
        
        raise IOError(f"Failed to obtain a valid model file after {max_retries} attempts.")


    def _checker_dm_limt(self, dm_limt):
        if isinstance(dm_limt, list):
            for item in dm_limt:
                if not isinstance(item, dict) or not all(
                    key in item for key in ["name", "dm_low", "dm_high"]
                ):
                    raise ValueError("Invalid format for dm_limt in config file.")
                if not isinstance(item["name"], str):
                    raise ValueError("Name in dm_limt must be a string.")
                if not all(
                    isinstance(item[key], (int, float)) for key in ["dm_low", "dm_high"]
                ):
                    raise ValueError("dm_low and dm_high in dm_limt must be numbers.")
        else:
            raise ValueError("Invalid format for dm_limt in config file.")

    def __str__(self):
        return str(self._config_data)

    @property
    def snrhold(self):
        snrhold = self._config_data.get("snrhold")
        if snrhold is None:
            snrhold = -100
        if not isinstance(snrhold, (int, float)):
            raise ValueError("snrhold must be a number.")
        return snrhold

    @property
    def modelname(self):
        modelnamedict = {
            "center-net": CENTERNET,
            "yolov11n": YOLOV11N,
            "detect-net": DETECTNET,
            "combine-net": COMBINENET,
        }
        
        modelname = self._config_data.get("modelname")
        if modelname is None:
            raise ValueError("modelname not found in config file.")
        if not isinstance(modelname, str):
            raise ValueError("modelname must be a string.")
        if modelname not in modelnamedict:
            raise ValueError(
                f"modelname must be one of {list(modelnamedict.keys())}, got {modelname}."
            )
        return modelnamedict[modelname]
    
    @property
    def maskfile(self):
        maskfile = self._config_data.get("maskfile")
        if maskfile is None:
            return ""
        if not isinstance(maskfile, str):
            raise ValueError("maskfile must be a string.")
        if not os.path.exists(maskfile):
            raise FileNotFoundError(f"Mask file {maskfile} does not exist.")
        return maskfile

    @property
    def mode(self):
        MODE = ["single", "directory", "muti", "monitor", "dataset"]
        mode = self._config_data.get("mode")
        if mode is None:
            raise ValueError(f"mode not found in config file. mode must be one of {MODE}.")
        if not isinstance(mode, str):
            raise ValueError("mode must be a string.")
        if mode not in MODE:
            raise ValueError(f"mode must be one of {MODE}, got {mode}.")
        return mode

    @property
    def candpath(self):
        if self.mode == "dataset":
            candpath = self._config_data.get("candpath")
            if candpath is None:
                raise ValueError("candpath not found in config file.")
            if candpath is not None and not isinstance(candpath, str):
                raise ValueError("candpath must be a string.")
            if not os.path.exists(candpath):
                raise FileNotFoundError(f"candpath {candpath} does not exist.")
            return candpath
        raise ValueError(f"mode is {self.mode} not `dataset`.")

    @property
    def modelpath(self):
        # https get
        modelpath = self._config_data.get("modelpath")
        if modelpath is None:
            modelpath = self.get_model()
        if not isinstance(modelpath, str):
            raise ValueError("modelpath must be a string.")
        if not os.path.exists(modelpath):
            raise FileNotFoundError(f"Model path {modelpath} does not exist.")
        return modelpath

    @property
    def dedgpu(self):
        return self._config_data.get("dedgpu", 0)

    @property
    def detgpu(self):
        return self._config_data.get("detgpu", 0)
    
    @property
    def include_last(self):
        include_last = self._config_data.get("include_last")
        if include_last is None:
            include_last = False
        if not isinstance(include_last, bool):
            raise ValueError("include_last must be a boolean.")
        return include_last

    @property
    def rficonfig(self):
        if self._rficonfig is None:
            rfi = self._config_data.get("rfi")
            if rfi is None:
                self._rficonfig = RFIConfig(use_mask=False, use_zero_dm=False, use_iqrm=False, iqrm_cfg=IQRMConfig())
                return self._rficonfig
        
            #rfi: 000 # use_mask:0 use_iqrm:0 use_zero_dm:0 dont use any rfi mitigation
            if not isinstance(rfi, dict):
                raise ValueError("rfi must be a dictionary.")
            use_mask = rfi.get("use_mask", False)
            use_zero_dm = rfi.get("use_zero_dm", False)
            use_iqrm = rfi.get("use_iqrm", False)

            iqrm_cfg = self.iqrmcfg if use_iqrm else IQRMConfig()
            self._rficonfig = RFIConfig(use_mask=use_mask, use_zero_dm=use_zero_dm, use_iqrm=use_iqrm, iqrm_cfg=iqrm_cfg)
        return self._rficonfig
    
    @property
    def iqrmcfg(self):
        if self._iqrmcfg is None:
            iqrmcfg = self._config_data.get("iqrm")
            if iqrmcfg is None:
                self._iqrmcfg = None
            else:
                if not isinstance(iqrmcfg, dict):
                    raise ValueError("iqrmcfg must be a dictionary.")
                required_keys = ["mode", "radius_frac", "nsigma", "geofactor", "win_sec", "hop_sec", "include_tail"]
                for key in required_keys:
                    if key not in iqrmcfg:
                        raise ValueError(f"iqrmcfg must contain the key '{key}'.")
                self._iqrmcfg = iqrmcfg
                self._iqrmcfg = IQRMConfig(
                    mode=iqrmcfg["mode"],
                    radius_frac=iqrmcfg["radius_frac"],
                    nsigma=iqrmcfg["nsigma"],
                    geofactor=iqrmcfg["geofactor"],
                    win_sec=iqrmcfg["win_sec"],
                    hop_sec=iqrmcfg["hop_sec"],
                    include_tail=iqrmcfg["include_tail"]
                )

        return self._iqrmcfg
    
    @property    
    def batchsize(self):
        batchsize = self._config_data.get("batchsize")
        if batchsize is None:
            batchsize = 128
        if not isinstance(batchsize, int):
            raise ValueError("batchsize must be an integer.")
        return batchsize
    
    @property
    def cputhread(self):
        return self._config_data.get("cputhread", 16)

    @property
    def plotworker(self):
        plotworker = self._config_data.get("plotworker")
        if plotworker is None:
            plotworker = 8
        if not isinstance(plotworker, int):
            raise ValueError("plotworker must be an integer.")
        return plotworker
    
    @property
    def maskdir(self):
        maskdir = self._config_data.get("maskdir")
        if maskdir is None:
            return None
        if not isinstance(maskdir, str):
            raise ValueError("maskdir must be a string.")
        if not os.path.exists(maskdir):
            raise FileNotFoundError(f"Mask directory {maskdir} does not exist.")
        return maskdir
    
    @property
    def dmtconfig(self):
        dmtconfig = self._config_data.get("dmtconfig")
        if dmtconfig is None:
            raise ValueError("dmtconfig not found in config file.")
        return dmtconfig

    @property
    def specconfig(self):
        specconfig = self._config_data.get("specconfig")
        if specconfig is None:
            raise ValueError("specconfig not found in config file.")
        return specconfig

    @property
    def dm_limt(self):
        dm_limt = self._config_data.get("dm_limt")
        if dm_limt is None:
            return None
        self._checker_dm_limt(dm_limt)
        return dm_limt

    @property
    def dmrange(self):
        dmrange = self._config_data.get("dmrange")
        if dmrange is None:
            raise ValueError("dmrange not found in config file.")
        self._checker_dmrange(dmrange)
        return dmrange

    @property
    def tsample(self):
        tsample = self._config_data.get("tsample")
        if tsample is None:
            raise ValueError("tsample not found in config file.")
        self._checker_tsample(tsample)
        return tsample

    @property
    def freqrange(self):
        freqrange = self._config_data.get("freqrange")
        if freqrange is None:
            raise ValueError("freqrange not found in config file.")
        self._checker_freqrange(freqrange)
        return freqrange

    @property
    def preprocess(self):
        preprocess = self._config_data.get("preprocess")
        if preprocess is None:
            return []
        return preprocess

    @property
    def input(self):
        return self._config_data.get("input")

    @property
    def output(self):
        path =  self._config_data.get("output")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    @property
    def timedownfactor(self):
        return self._config_data.get("timedownfactor")

    @property
    def confidence(self):
        return self._config_data.get("confidence")
