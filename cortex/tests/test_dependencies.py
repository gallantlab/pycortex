"""Tests for cortex/dependencies.py"""
import subprocess
from unittest import mock

import pytest

from cortex.dependencies import (
    _query_inkscape_version,
    inkscape_path,
    inkscape_version,
    require_inkscape,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    _query_inkscape_version.cache_clear()
    yield
    _query_inkscape_version.cache_clear()


def _make_run_result(stdout=b'', stderr=b'', returncode=0):
    """Helper to build a mock subprocess.CompletedProcess."""
    result = mock.MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


@mock.patch('cortex.dependencies.inkscape_path', return_value=None)
def test_inkscape_version_not_installed(mock_path):
    """Returns None when inkscape is neither configured nor on PATH."""
    assert inkscape_version() is None


@mock.patch('cortex.dependencies.inkscape_path', return_value='inkscape')
@mock.patch('cortex.dependencies.sp.run')
def test_inkscape_version_clean_stdout(mock_run, mock_path):
    """Parses version correctly from clean stdout output."""
    mock_run.return_value = _make_run_result(
        stdout=b'Inkscape 1.0 (4035a4f, 2020-05-01)\n'
    )
    assert inkscape_version() == '1.0'


@mock.patch('cortex.dependencies.inkscape_path', return_value='inkscape')
@mock.patch('cortex.dependencies.sp.run')
def test_inkscape_version_newer(mock_run, mock_path):
    """Parses a newer version number correctly."""
    mock_run.return_value = _make_run_result(
        stdout=b'Inkscape 1.2.2 (b0a8486, 2022-12-01)\n'
    )
    assert inkscape_version() == '1.2.2'


@mock.patch('cortex.dependencies.inkscape_path', return_value='inkscape')
@mock.patch('cortex.dependencies.sp.run')
def test_inkscape_version_with_diagnostic_noise(mock_run, mock_path):
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


@mock.patch('cortex.dependencies.inkscape_path', return_value='inkscape')
@mock.patch('cortex.dependencies.sp.run')
def test_inkscape_version_noise_in_stderr(mock_run, mock_path):
    """Returns correct version when the diagnostic message is on stderr."""
    mock_run.return_value = _make_run_result(
        stderr=b'Setting _INKSCAPE_GC=disable as a workaround for broken libgc\n',
        stdout=b'Inkscape 1.2.2 (b0a8486, 2022-12-01)\n',
    )
    assert inkscape_version() == '1.2.2'


@mock.patch('cortex.dependencies.inkscape_path', return_value='inkscape')
@mock.patch('cortex.dependencies.sp.run')
def test_inkscape_version_no_inkscape_line(mock_run, mock_path):
    """Returns None when no 'Inkscape …' line is found in output."""
    mock_run.return_value = _make_run_result(stdout=b'some unexpected output\n')
    assert inkscape_version() is None


@mock.patch('cortex.dependencies.inkscape_path', return_value='inkscape')
@mock.patch('cortex.dependencies.sp.run')
def test_inkscape_version_subprocess_error(mock_run, mock_path):
    """Propagates CalledProcessError when inkscape exits non-zero."""
    mock_run.side_effect = subprocess.CalledProcessError(1, 'inkscape')
    with pytest.raises(subprocess.CalledProcessError):
        inkscape_version()


@mock.patch('cortex.dependencies.sp.run')
@mock.patch('cortex.dependencies.which')
@mock.patch('cortex.dependencies.config')
def test_inkscape_lookup_retries_until_found(mock_config, mock_which, mock_run):
    """A failed lookup is not cached, so a later install is picked up."""
    mock_config.get.return_value = ''
    mock_which.return_value = None
    assert inkscape_path() is None
    assert inkscape_version() is None

    # Inkscape gets installed while the process is still running.
    mock_which.return_value = '/usr/bin/inkscape'
    mock_run.return_value = _make_run_result(
        stdout=b'Inkscape 1.2.2 (b0a8486, 2022-12-01)\n'
    )
    assert inkscape_path() == '/usr/bin/inkscape'
    assert inkscape_version() == '1.2.2'

    # Once found, the version is cached and the subprocess is not run again.
    mock_run.reset_mock()
    assert inkscape_version() == '1.2.2'
    mock_run.assert_not_called()


@mock.patch('cortex.dependencies.inkscape_path', return_value=None)
def test_require_inkscape_missing(mock_path):
    """Raises an error that points at the config option when inkscape is missing."""
    with pytest.raises(RuntimeError, match=r'\[dependency_paths\]'):
        require_inkscape()


@mock.patch('cortex.dependencies.inkscape_path', return_value='/usr/bin/inkscape')
@mock.patch('cortex.dependencies.sp.run')
def test_require_inkscape_found(mock_run, mock_path):
    """Returns the path and version together when inkscape is available."""
    mock_run.return_value = _make_run_result(
        stdout=b'Inkscape 1.2.2 (b0a8486, 2022-12-01)\n'
    )
    assert require_inkscape() == ('/usr/bin/inkscape', '1.2.2')
