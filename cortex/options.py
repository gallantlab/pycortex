import os
try:
    import configparser
except ImportError:
    import ConfigParser as configparser
from . import appdirs

cwd = os.path.split(os.path.abspath(__file__))[0]
userdir = appdirs.user_data_dir("pycortex", "JamesGao")
usercfg = os.path.join(userdir, "options.cfg")

config = configparser.ConfigParser()
config.readfp(open(os.path.join(cwd, 'defaults.cfg')))

if len(config.read(usercfg)) == 0:
    os.makedirs(userdir)
    with open(usercfg, 'w') as fp:
        config.write(fp)
        
#set default path in case the module is imported from the source code directory
if not config.has_option("basic", "filestore"):
    config.set("basic", "filestore", os.path.join(cwd, os.pardir, "filestore/db"))

if not config.has_option("webgl", "colormaps"):
    config.set("webgl", "colormaps", os.path.join(cwd, os.pardir, "filestore/colormaps"))
