"""Tests for the viewer plumbing in cortex.webgl.view that needs no browser."""

import pytest

from cortex.webgl import serve, view


def test_local_url_is_localhost(monkeypatch):
    """The URL opened on this machine must not depend on the hostname.

    socket.gethostname() returns the mDNS name on macOS, which resolves to a
    list of addresses (link-local ones included) and only reaches the server if
    the firewall lets python accept connections on a non-loopback interface.
    """
    monkeypatch.setattr(serve, "hostname", "mymac.local")
    monkeypatch.setattr(view, "domain_name", "")

    local, network = view._viewer_urls(39140)
    assert local == "http://localhost:39140/mixer.html"
    assert network == "http://mymac.local:39140/mixer.html"


def test_network_url_appends_configured_domain(monkeypatch):
    """webgl.domain_name is appended to the hostname, and never to localhost."""
    monkeypatch.setattr(serve, "hostname", "mymac")
    monkeypatch.setattr(view, "domain_name", ".example.org")

    local, network = view._viewer_urls(8080)
    assert local == "http://localhost:8080/mixer.html"
    assert network == "http://mymac.example.org:8080/mixer.html"


@pytest.mark.parametrize("hostname", ["localhost", "mymac.local", "mymac"])
def test_urls_agree_on_port_and_path(monkeypatch, hostname):
    """Both URLs address the same running server."""
    monkeypatch.setattr(serve, "hostname", hostname)
    monkeypatch.setattr(view, "domain_name", "")

    local, network = view._viewer_urls(1234)
    for url in (local, network):
        assert url.startswith("http://")
        assert url.endswith(":1234/mixer.html")
