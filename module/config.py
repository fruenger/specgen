try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
    
import os

def load_config(filename="config.toml"):
    
    module_dir  = os.path.dirname(__file__)
    
    with open(os.path.join(module_dir, "config.toml"), mode="rb") as fp:
        config = tomllib.load(fp)
    
    return config
