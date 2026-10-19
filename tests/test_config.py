import os
from pathlib import Path

import pytest

from laya_strix.config import Settings


def _settings(tmp_path, **kwargs):
    base = dict(
        model_dir=tmp_path,
        encoder=tmp_path / "encoder.onnx",
        head=tmp_path / "head.onnx",
    )
    base.update(kwargs)
    return Settings(**base)


def test_api_key_file_is_read_and_applied(tmp_path, monkeypatch):
    key_file = tmp_path / "api.key"
    key_file.write_text("file-key\n")
    key_file.chmod(0o600)
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    settings = _settings(tmp_path, api_key_file=key_file)

    assert settings.resolve_api_key() == "file-key"
    assert settings.apply_api_key() == "file-key"
    assert os.environ["LAYA_API_KEY"] == "file-key"


def test_env_key_wins_over_file(tmp_path, monkeypatch):
    key_file = tmp_path / "api.key"
    key_file.write_text("file-key\n")
    key_file.chmod(0o600)
    monkeypatch.setenv("LAYA_API_KEY", "env-key")
    settings = _settings(tmp_path, api_key_file=key_file)

    assert settings.resolve_api_key() == "env-key"


def test_empty_key_file_fails_closed(tmp_path, monkeypatch):
    key_file = tmp_path / "api.key"
    key_file.write_text("\n")
    key_file.chmod(0o600)
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    settings = _settings(tmp_path, api_key_file=key_file)

    with pytest.raises(RuntimeError, match="empty"):
        settings.resolve_api_key()


def test_tls_requires_both_cert_and_key(tmp_path):
    cert = tmp_path / "server.crt"
    cert.touch()
    settings = _settings(tmp_path, tls_cert=cert)

    with pytest.raises(RuntimeError, match="both"):
        settings.validate_tls()


def test_tls_validates_files_and_client_ca(tmp_path):
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    ca = tmp_path / "ca.crt"
    cert.touch()
    key.write_text("dummy-key")
    key.chmod(0o600)
    ca.touch()

    _settings(tmp_path, tls_cert=cert, tls_key=key).validate_tls()
    _settings(tmp_path, tls_cert=cert, tls_key=key, tls_ca=ca,
              tls_client_auth=True).validate_tls()

    with pytest.raises(RuntimeError, match="requires LAYA_TLS_CA"):
        _settings(tmp_path, tls_cert=cert, tls_key=key,
                  tls_client_auth=True).validate_tls()

    missing = tmp_path / "nope.pem"
    with pytest.raises(RuntimeError, match="does not exist"):
        _settings(tmp_path, tls_cert=missing, tls_key=key).validate_tls()


def test_describe_does_not_include_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "secret-value")
    settings = _settings(tmp_path)
    described = settings.describe()

    assert described["api_key_configured"] is True
    assert "secret-value" not in repr(described)


def test_api_key_file_permissions_are_checked(tmp_path, monkeypatch):
    key_file = tmp_path / "api.key"
    key_file.write_text("file-key\n")
    key_file.chmod(0o644)
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    settings = _settings(tmp_path, api_key_file=key_file)

    with pytest.raises(RuntimeError, match="chmod 600"):
        settings.resolve_api_key()


def test_numeric_settings_are_validated(tmp_path):
    with pytest.raises(RuntimeError, match="LAYA_HEAD_THREADS"):
        _settings(tmp_path, head_threads=0).validate()
    with pytest.raises(RuntimeError, match="LAYA_SEQ_LEN"):
        _settings(tmp_path, seq_len=0).validate()
    with pytest.raises(RuntimeError, match="LAYA_PORT"):
        _settings(tmp_path, port=70000).validate()


def test_whitespace_env_key_is_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "   ")
    settings = _settings(tmp_path)
    assert settings.apply_api_key() is None
    assert "LAYA_API_KEY" not in os.environ
