import os
import yaml

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

    def _load_config(self):
        if not self.config_file or not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file {self.config_file} not found.")
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)
        
    def _checker_dmrange(self, dmrange):
        if isinstance(dmrange, dict):
            for item in dmrange:
                if not all(key in item for key in ["name", "dm_low", "dm_high", "dm_step"]):
                    raise ValueError("Invalid format for dmrange in config file.")
                if not isinstance(item["name"], str):
                    raise ValueError("Name must be a string.")
                if not all(isinstance(item[key], (int, float)) for key in ["dm_low", "dm_high", "dm_step"]):
                    raise ValueError("dm_low, dm_high, and dm_step must be numbers.")
        else:
            raise ValueError("Invalid format for dmrange in config file.")

    def __str__(self):
        return str(self._config_data)

    @property
    def dmrange(self):
        dmrange = self._config_data.get("dmrange", None)
        dms = []
        if dmrange is not None:
            self._checker_dmrange(dmrange)
            return dmrange
        else:
            raise ValueError("dmrange not found in config file.")
        return self._config_data.get("dmrange", None)
    
    @property
    def dmlimt(self):
        return self._config_data.get("dmlimt", None)
    
    @property
    def tsample(self):
        return self._config_data.get("tsample", None)
    
    @property
    def freqrange(self):
        return self._config_data.get("freqrange", None)
    
    @property
    def preprocess(self):
        return self._config_data.get("preprocess", None)
    
    @property
    def inputdir(self):
        return self._config_data.get("inputdir", None)
    
    @property
    def outputdir(self):
        return self._config_data.get("outputdir", None)

