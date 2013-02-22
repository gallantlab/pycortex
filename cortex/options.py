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