import os
from sympy import preorder_traversal
import yaml

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
        self.config_file = config_file
        self._config_data = self._load_config()
        cputhread = self._config_data.get("cputhread", 16)
        os.environ['OMP_NUM_THREADS'] = f'{cputhread}'
        
    def _load_config(self):
        if not self.config_file or not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file {self.config_file} not found.")
        with open(self.config_file, "r") as file:
            return yaml.safe_load(file)

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

    def _checker_preprocess(self, preprocess):
        if isinstance(preprocess, list):
            for item in preprocess:
                if not isinstance(item, dict) or len(item) != 1:
                    raise ValueError(
                        "Invalid format for preprocess in config file. Each item should be a single key-value pair."
                    )
        else:
            raise ValueError("Invalid format for preprocess in config file.")

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
            snrhold = 0
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
            raise ValueError("mode not found in config file. mode must be one of {MODE}.")
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
    def modepath(self):
        modepath = self._config_data.get("modepath")
        if modepath is None:
            raise ValueError("modepath not found in config file.")
        if not isinstance(modepath, str):
            raise ValueError("modepath must be a string.")
        if not os.path.exists(modepath):
            raise FileNotFoundError(f"Mode path {modepath} does not exist.")
        return modepath


    @property
    def dedgpu(self):
        return self._config_data.get("dedgpu", 0)

    @property
    def detgpu(self):
        return self._config_data.get("detgpu", 0)
    
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
            raise ValueError("dm_limt not found in config file.")
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
