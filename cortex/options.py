import os

from six.moves import configparser
import six

if six.PY2:
  ConfigParser = configparser.SafeConfigParser
else:
  ConfigParser = configparser.ConfigParser

from . import appdirs

def add_defaults(config, defaults):
    """Adds default values to a configuration

    Checks all entries of a default configuration and adds them to another
    configuration if it does not have a value for that entry.

    Parameters
    ----------
    config : configparser.ConfigParser
    defaults : configparser.ConfigParser

    Returns
    -------
    Boolean indicating whether or not configuration was changed.
    """
    updated = False
    for section in defaults.sections():
        if not config.has_section(section):
            updated = True
            config.add_section(section)

        for item, default in defaults.items(section):
            try:
                value = config.get(section, item)
            except (configparser.NoOptionError):
                updated = True
                config.set(section, item, default)

    return updated

cwd = os.path.split(os.path.abspath(__file__))[0]

# read default options
default_path = os.path.join(cwd, 'defaults.cfg')
defaults = ConfigParser()
_ = defaults.read(default_path)

# read user options
config_dir = appdirs.user_data_dir("pycortex", "JamesGao")
config_path = os.path.join(config_dir, "options.cfg")
config = ConfigParser()

if os.path.exists(config_path):
    _ = config.read(config_path)
else:
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

# overwrite defaults with user values
updated = add_defaults(config, defaults)

if updated:
    print ('updating user configuration file at {}'.format(config_path))
    # write updated configuration to user config file
    with open(config_path, 'w') as config_file:
        config.write(config_file)

# set default path in case the module is imported from the source code
# directory
if not config.has_option("basic", "filestore"):
    config.set("basic", "filestore",
               os.path.join(cwd, os.pardir, "filestore/db"))

if not config.has_option("webgl", "colormaps"):
    config.set("webgl", "colormaps",
               os.path.join(cwd, os.pardir, "filestore/colormaps"))
