import os
import configparser
from . import appdirs

cwd = os.path.split(os.path.abspath(__file__))[0]
userdir = appdirs.user_data_dir("pycortex")
usercfg = os.path.join(userdir, "options.cfg")

# Read defaults from pycortex repo
config = configparser.ConfigParser()
config.read(os.path.join(cwd, 'defaults.cfg'))

# Update defaults with user-sepecifed values in user config
files_successfully_read = config.read(usercfg)

# If user config doesn't exist, create it
if len(files_successfully_read) == 0:
    if not os.path.exists(userdir):
        os.makedirs(userdir)
    with open(usercfg, 'w') as fp:
        config.write(fp)
        
#set default path in case the module is imported from the source code directory
if not config.has_option("basic", "filestore"):
    config.set("basic", "filestore", os.path.join(cwd, os.pardir, "filestore/db"))

if not config.has_option("webgl", "colormaps"):
    config.set("webgl", "colormaps", os.path.join(cwd, os.pardir, "filestore/colormaps"))


def set_user_option(section: str, option: str, value: str) -> str:
    """Persist a single option to the user config file.

    Updates the in-process ``config`` as well, so the new value takes effect
    immediately without a reimport.

    Only the user config is rewritten, and only the keys it already contained
    plus this one. The module-level ``config`` is the *merge* of
    ``defaults.cfg`` and the user config (plus a couple of source-checkout
    fallbacks patched in above), so writing it out wholesale would freeze every
    current default -- including interpreter-relative filestore paths -- into
    the user's file.

    Parameters
    ----------
    section : str
        Config section, e.g. ``'webshow'``. Created if absent.
    option : str
        Option name within `section`.
    value : str
        Value to store. Config values are always strings.

    Returns
    -------
    str
        Path of the user config file that was written.
    """
    user_config = configparser.ConfigParser()
    user_config.read(usercfg)
    if not user_config.has_section(section):
        user_config.add_section(section)
    user_config.set(section, option, value)

    os.makedirs(userdir, exist_ok=True)
    with open(usercfg, 'w') as fp:
        user_config.write(fp)

    if not config.has_section(section):
        config.add_section(section)
    config.set(section, option, value)
    return usercfg
