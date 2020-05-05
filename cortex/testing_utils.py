"""Module containing utils for testing"""
from shutil import which


def has_installed(name):
    return which(name) is not None

