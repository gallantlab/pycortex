import os
import ConfigParser
import appdirs

userdir = appdirs.user_data_dir("pycortex", "JamesGao")
usercfg = os.path.join(userdir, "options.cfg")

config = ConfigParser.ConfigParser()
config = config.readfp(open('defaults.cfg'))
config.read(usercfg)

def set_default_filestore(path):
    store = os.path.expanduser('~/pycortex_store/')
    config.set("basic", "filestore", store)
    with open(path, 'w') as fp:
        config.write(fp)
