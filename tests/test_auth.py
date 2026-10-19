from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from laya_strix.auth import _expected, make_api_key_dependency


def _app(api_key):
    app = FastAPI()
    dependency = make_api_key_dependency(api_key)

    @app.get("/private", dependencies=[Depends(dependency)])
    def private():
        return {"ok": True}

    return TestClient(app)


def test_no_key_is_open():
    client = _app(None)
    assert client.get("/private").status_code == 200


def test_missing_key_is_401():
    response = _app("s3cret").get("/private")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_wrong_key_is_401():
    response = _app("s3cret").get(
        "/private", headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401


def test_correct_key_is_200():
    response = _app("s3cret").get(
        "/private", headers={"Authorization": "Bearer s3cret"})
    assert response.status_code == 200


def test_non_ascii_key_matches_wire_encoding():
    # HTTP clients generally refuse non-ASCII header values; laya.serve
    # deliberately uses surrogateescape for clients that put the wire bytes
    # through.  Check the encoded value rather than asking TestClient to send
    # an invalid ASCII header.
    key = "s\u00e9cret"
    assert _expected(key) == ("Bearer " + key).encode("utf-8", "surrogateescape")
