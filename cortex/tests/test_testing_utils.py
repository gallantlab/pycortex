"""Tests for cortex/testing_utils.py"""
import subprocess
from unittest import mock

import pytest

from cortex.testing_utils import inkscape_version


def _make_run_result(stdout=b'', stderr=b'', returncode=0):
    """Helper to build a mock subprocess.CompletedProcess."""
    result = mock.MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


@mock.patch('cortex.testing_utils.has_installed', return_value=False)
def test_inkscape_version_not_installed(mock_has):
    """Returns None when inkscape is not on PATH."""
    assert inkscape_version() is None


@mock.patch('cortex.testing_utils.has_installed', return_value=True)
@mock.patch('cortex.testing_utils.sp.run')
def test_inkscape_version_clean_stdout(mock_run, mock_has):
    """Parses version correctly from clean stdout output."""
    mock_run.return_value = _make_run_result(
        stdout=b'Inkscape 1.0 (4035a4f, 2020-05-01)\n'
    )
    assert inkscape_version() == '1.0'


@mock.patch('cortex.testing_utils.has_installed', return_value=True)
@mock.patch('cortex.testing_utils.sp.run')
def test_inkscape_version_newer(mock_run, mock_has):
    """Parses a newer version number correctly."""
    mock_run.return_value = _make_run_result(
        stdout=b'Inkscape 1.2.2 (b0a8486, 2022-12-01)\n'
    )
    assert inkscape_version() == '1.2.2'


@mock.patch('cortex.testing_utils.has_installed', return_value=True)
@mock.patch('cortex.testing_utils.sp.run')
def test_inkscape_version_with_diagnostic_noise(mock_run, mock_has):
    """Returns correct version even when a diagnostic message precedes it.

    This is the regression test for the bug where systems that print
    "Setting _INKSCAPE_GC=disable as a workaround for broken libgc"
    caused INKSCAPE_VERSION to be set to '_INKSCAPE_GC=disable'.
    """
    mock_run.return_value = _make_run_result(
        stdout=(
            b'Setting _INKSCAPE_GC=disable as a workaround for broken libgc\n'
            b'Inkscape 1.2.2 (b0a8486, 2022-12-01)\n'
        )
    )
    assert inkscape_version() == '1.2.2'


@mock.patch('cortex.testing_utils.has_installed', return_value=True)
@mock.patch('cortex.testing_utils.sp.run')
def test_inkscape_version_noise_in_stderr(mock_run, mock_has):
    """Returns correct version when the diagnostic message is on stderr."""
    mock_run.return_value = _make_run_result(
        stderr=b'Setting _INKSCAPE_GC=disable as a workaround for broken libgc\n',
        stdout=b'Inkscape 1.2.2 (b0a8486, 2022-12-01)\n',
    )
    assert inkscape_version() == '1.2.2'


@mock.patch('cortex.testing_utils.has_installed', return_value=True)
@mock.patch('cortex.testing_utils.sp.run')
def test_inkscape_version_no_inkscape_line(mock_run, mock_has):
    """Returns None when no 'Inkscape …' line is found in output."""
    mock_run.return_value = _make_run_result(stdout=b'some unexpected output\n')
    assert inkscape_version() is None


@mock.patch('cortex.testing_utils.has_installed', return_value=True)
@mock.patch('cortex.testing_utils.sp.run')
def test_inkscape_version_subprocess_error(mock_run, mock_has):
    """Propagates CalledProcessError when inkscape exits non-zero."""
    mock_run.side_effect = subprocess.CalledProcessError(1, 'inkscape')
    with pytest.raises(subprocess.CalledProcessError):
        inkscape_version()
