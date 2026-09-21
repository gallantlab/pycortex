#!/usr/bin/env python
# Copyright (c) 2005-2010 ActiveState Software Inc.

"""Utilities for determining application-specific dirs.

See <http://github.com/ActiveState/appdirs> for details and usage.
"""
# Dev Notes:
# - MSDN on where to store app data files:
#   http://support.microsoft.com/default.aspx?scid=kb;en-us;310294#XSLTH3194121123120121120120
# - Mac OS X: http://developer.apple.com/documentation/MacOSX/Conceptual/BPFileSystem/index.html
# - XDG spec for Un*x: http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html

__version_info__ = (1, 2, 0)
__version__ = '.'.join([str(x) for x in __version_info__])


import sys
import os

PY3 = sys.version_info[0] == 3

if PY3:
    str = str

def user_data_dir(appname, version=None, roaming=False):
    r"""Return full path to the user-specific data dir for this application.

        "appname" is the name of application.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
        "roaming" (boolean, default False) can be set True to use the Windows
            roaming appdata directory. That means that for users on a Windows
            network setup for roaming profiles, this user data will be
            sync'd on login. See
            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>
            for a discussion of issues.

    Typical user data directories are:
        Mac OS X:               ~/Library/Application Support/<AppName>
        Unix:                   ~/.config/<appname>    # or in $XDG_CONFIG_HOME if defined
        Windows (not roaming):  C:\Users\<username>\AppData\Local\<AppName>
        Windows (roaming):      C:\Users\<username>\AppData\Roaming\<AppName>

    For Unix, we follow the XDG spec and support $XDG_CONFIG_HOME. We don't
    use $XDG_DATA_HOME as that data dir is mostly used at the time of
    installation, instead of the application adding data during runtime.
    Also, in practice, Linux apps tend to store their data in
    "~/.config/<appname>" instead of "~/.local/share/<appname>".
    """
    if sys.platform.startswith("win"):
        environmentVariableName = "APPDATA" if roaming else "LOCALAPPDATA"
        path = os.path.join(os.environ[environmentVariableName], appname)
    elif sys.platform == 'darwin':
        path = os.path.join(
            os.path.expanduser('~/Library/Application Support/'),
            appname)
    else:
        path = os.path.join(
            os.getenv('XDG_CONFIG_HOME', os.path.expanduser("~/.config")),
            appname.lower())
    if version:
        path = os.path.join(path, version)
    return path


def site_data_dir(appname, version=None):
    """Return full path to the user-shared data dir for this application.

        "appname" is the name of application.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".

    Typical user data directories are:
        Mac OS X:   /Library/Application Support/<AppName>
        Unix:       /etc/xdg/<appname>
        Windows:    C:\\ProgramData\\<AppName>

    For Unix, this is using the $XDG_CONFIG_DIRS[0] default.
    """
    if sys.platform.startswith("win"):
        path = os.path.join(os.environ["PROGRAMDATA"], appname)
    elif sys.platform == 'darwin':
        path = os.path.join(
            os.path.expanduser('/Library/Application Support'),
            appname)
    else:
        # XDG default for $XDG_CONFIG_DIRS[0]. Perhaps should actually
        # *use* that envvar, if defined.
        path = "/etc/xdg/"+appname.lower()
    if version:
        path = os.path.join(path, version)
    return path


def user_cache_dir(appname, version=None):
    r"""Return full path to the user-specific cache dir for this application.

        "appname" is the name of application.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".

    Typical user cache directories are:
        Mac OS X:   ~/Library/Caches/<AppName>
        Unix:       ~/.cache/<appname> (XDG default)
        Windows:    C:\Users\<username>\AppData\Local\<AppName>\Cache

    On Windows local settings go in the same directory as the non-roaming
    app data dir (the default returned by `user_data_dir` above). Apps typically
    put cache data somewhere *under* the given dir here. Some examples:
        ...\Mozilla\Firefox\Profiles\<ProfileName>\Cache
        ...\Acme\SuperApp\Cache\1.0
    This function appends "Cache" to the local app data value.
    """
    if sys.platform.startswith("win"):
        path = os.path.join(os.environ["LOCALAPPDATA"], appname, "Cache")
    elif sys.platform == 'darwin':
        path = os.path.join(
            os.path.expanduser('~/Library/Caches'),
            appname)
    else:
        path = os.path.join(
            os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache')),
            appname.lower())
    if version:
        path = os.path.join(path, version)
    return path

def user_log_dir(appname, version=None):
    r"""Return full path to the user-specific log dir for this application.

        "appname" is the name of application.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".

    Typical user cache directories are:
        Mac OS X:   ~/Library/Logs/<AppName>
        Unix:       ~/.cache/<appname>/log  # or under $XDG_CACHE_HOME if defined
        Windows:    C:\Users\<username>\AppData\Local\<AppName>\Logs

    This function appends "Logs" to the local app data
    value for Windows and appends "log" to the user cache dir for Unix.
    """
    if sys.platform == "darwin":
        path = os.path.join(
            os.path.expanduser('~/Library/Logs'),
            appname)
    elif sys.platform == "win32":
        path = user_data_dir(appname, version); version=False
        path = os.path.join(path, "Logs")
    else:
        path = user_cache_dir(appname, version); version=False
        path = os.path.join(path, "log")
    if version:
        path = os.path.join(path, version)
    return path


class AppDirs:
    """Convenience wrapper for getting application dirs."""
    def __init__(self, appname, version=None, roaming=False):
        self.appname = appname
        self.version = version
        self.roaming = roaming
    @property
    def user_data_dir(self):
        return user_data_dir(self.appname,
            version=self.version, roaming=self.roaming)
    @property
    def site_data_dir(self):
        return site_data_dir(self.appname,
            version=self.version)
    @property
    def user_cache_dir(self):
        return user_cache_dir(self.appname,
            version=self.version)
    @property
    def user_log_dir(self):
        return user_log_dir(self.appname,
            version=self.version)




#---- self test code

if __name__ == "__main__":
    appname = "MyApp"

    props = ("user_data_dir", "site_data_dir", "user_cache_dir",
        "user_log_dir")

    print("-- app dirs (without optional 'version')")
    dirs = AppDirs(appname, version="1.0")
    for prop in props:
        print(("%s: %s" % (prop, getattr(dirs, prop))))

    print("\n-- app dirs (with optional 'version')")
    dirs = AppDirs(appname)
    for prop in props:
        print(("%s: %s" % (prop, getattr(dirs, prop))))
