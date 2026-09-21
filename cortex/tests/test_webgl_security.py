"""Tests for the pre-server security prompt in ``cortex.webgl.security``."""

import io

import pytest

from cortex import options
from cortex.webgl import security


@pytest.fixture
def config_flag():
    """Save and restore the config option the prompt reads and writes."""
    had = options.config.has_option(security.CONFIG_SECTION, security.CONFIG_OPTION)
    old = (
        options.config.get(security.CONFIG_SECTION, security.CONFIG_OPTION)
        if had
        else None
    )
    yield
    if had:
        options.config.set(security.CONFIG_SECTION, security.CONFIG_OPTION, old)
    else:
        options.config.remove_option(
            security.CONFIG_SECTION, security.CONFIG_OPTION
        )


@pytest.fixture
def interactive(monkeypatch, config_flag):
    """Undo the suite-wide opt-out so the prompt actually runs."""
    monkeypatch.delenv(security.ENV_VAR, raising=False)
    options.config.set(security.CONFIG_SECTION, security.CONFIG_OPTION, "false")


def answer(monkeypatch, *responses):
    """Feed `responses` to the prompt, one per call to ``input()``."""
    replies = iter(responses)

    def fake_input():
        return next(replies)

    monkeypatch.setattr("builtins.input", fake_input)


def test_env_var_disables_warning(monkeypatch):
    monkeypatch.setenv(security.ENV_VAR, "1")
    assert security.warning_is_disabled()

    stream = io.StringIO()
    security.confirm_server_start(stream=stream)
    assert stream.getvalue() == ""


def test_config_flag_disables_warning(monkeypatch, config_flag):
    monkeypatch.delenv(security.ENV_VAR, raising=False)
    options.config.set(security.CONFIG_SECTION, security.CONFIG_OPTION, "true")
    assert security.warning_is_disabled()

    stream = io.StringIO()
    security.confirm_server_start(stream=stream)
    assert stream.getvalue() == ""


def test_yes_starts_viewer(interactive, monkeypatch):
    answer(monkeypatch, "y")
    stream = io.StringIO()
    security.confirm_server_start(stream=stream)
    assert "SECURITY WARNING" in stream.getvalue()


def test_no_aborts(interactive, monkeypatch):
    answer(monkeypatch, "n")
    with pytest.raises(security.ViewerStartAborted):
        security.confirm_server_start(stream=io.StringIO())


def test_invalid_answer_reprompts(interactive, monkeypatch):
    answer(monkeypatch, "maybe", "", "y")
    stream = io.StringIO()
    security.confirm_server_start(stream=stream)
    assert stream.getvalue().count("Please answer") == 2


def test_ignore_writes_user_config(interactive, monkeypatch, tmp_path):
    usercfg = tmp_path / "options.cfg"
    monkeypatch.setattr(options, "userdir", str(tmp_path))
    monkeypatch.setattr(options, "usercfg", str(usercfg))
    answer(monkeypatch, "i")

    stream = io.StringIO()
    security.confirm_server_start(stream=stream)

    assert str(usercfg) in stream.getvalue()
    written = usercfg.read_text()
    assert f"{security.CONFIG_OPTION} = true" in written

    # the in-process config is updated too, so the warning is skipped for the
    # rest of this session without a reimport (config_flag restores it)
    assert options.config.getboolean(
        security.CONFIG_SECTION, security.CONFIG_OPTION
    )


def test_ignore_preserves_other_user_options(interactive, monkeypatch, tmp_path):
    usercfg = tmp_path / "options.cfg"
    usercfg.write_text("[basic]\nfilestore = /somewhere/private\n")
    monkeypatch.setattr(options, "userdir", str(tmp_path))
    monkeypatch.setattr(options, "usercfg", str(usercfg))
    answer(monkeypatch, "i")

    security.confirm_server_start(stream=io.StringIO())

    written = usercfg.read_text()
    assert "filestore = /somewhere/private" in written
    assert f"{security.CONFIG_OPTION} = true" in written


@pytest.mark.parametrize("error", [EOFError, OSError, RuntimeError])
def test_non_interactive_warns_and_continues(interactive, monkeypatch, error):
    def no_stdin():
        raise error("no console")

    monkeypatch.setattr("builtins.input", no_stdin)
    stream = io.StringIO()
    security.confirm_server_start(stream=stream)

    output = stream.getvalue()
    assert "SECURITY WARNING" in output
    assert security.ENV_VAR in output


def test_warning_names_the_working_directory():
    text = security.warning_text(cwd="/tmp/some-analysis")
    assert "/tmp/some-analysis" in text


def test_warning_names_the_movie_directory():
    text = security.warning_text(cwd="/tmp/cwd", movie_root="/tmp/frames")
    assert "/tmp/frames" in text


def test_movie_directory_defaults_to_the_working_directory():
    text = security.warning_text(cwd="/tmp/some-analysis")
    assert text.count("/tmp/some-analysis") == 2
