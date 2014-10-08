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

if not config.has_option("basic", "filestore"):
    config.set("webgl", "colormaps", os.path.join(cwd, os.pardir, "filestore/colormaps"))

# For backward compatibility with pre-sulci versions: 
if not config.has_section('sulci'):
    config.add_section('sulci')
    sulc = dict(line_width = '2',
                line_color = '1., 1., 1., 1.',
                fill_color = '0., 0., 0., 0.',
                shadow = '3',
                labelsize = '16pt',
                labelcolor = '1., 1., 1., 1.')
    for k,v in sulc.items():
        config.set('sulci',k,v)
    del k,v
    print('Automatically added sulci section to options.config...')
    # Write / warn that this has been added? 
    with open(usercfg, 'w') as fp:
        config.write(fp)
