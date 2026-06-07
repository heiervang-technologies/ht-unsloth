import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lile.middleware import RequestIDMiddleware, AuthMiddleware
from lile.server_errors import register_error_handlers

pytestmark = pytest.mark.cpu_only

def _build_app(api_key="secret", routes=["/v1/train", "/v1/state/*"]):
    app = FastAPI()
    app.add_middleware(AuthMiddleware, api_key=api_key, auth_required_routes=routes)
    app.add_middleware(RequestIDMiddleware)
    register_error_handlers(app)

    @app.get("/health")
    def health(): return {"ok": True}

    @app.post("/v1/chat/completions")
    def chat(): return {"ok": True}

    @app.post("/v1/train")
    def train(): return {"ok": True}
    
    @app.get("/v1/state/snapshots")
    def snaps(): return {"ok": True}

    return app

def test_auth_rejects_no_key():
    app = _build_app()
    with TestClient(app) as client:
        res = client.post("/v1/chat/completions", json={"messages": []})
        assert res.status_code == 200

        res = client.post("/v1/train", json={"objective": "sft"})
        assert res.status_code == 401
        assert res.json()["error"]["code"] == "unauthorized"

def test_auth_accepts_valid_key():
    app = _build_app()
    with TestClient(app) as client:
        res = client.post("/v1/train", json={"objective": "sft"}, headers={"Authorization": "Bearer secret"})
        assert res.status_code == 200

def test_auth_bypasses_public_routes():
    app = _build_app()
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        
        res = client.get("/v1/state/snapshots")
        assert res.status_code == 401

def test_auth_disabled():
    app = _build_app(api_key=None)
    with TestClient(app) as client:
        res = client.post("/v1/train")
        assert res.status_code == 200
