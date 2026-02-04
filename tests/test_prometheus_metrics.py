import pytest

pytest.importorskip("prometheus_client")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

import tgefd.server as server


def test_prometheus_metrics_endpoint():
    client = TestClient(server.app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "tgefd_http_requests_total" in response.text
