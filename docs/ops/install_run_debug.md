# Install / Run / Debug (Ops)

## Install
```bash
python -m pip install -e .[server]
```

## Run (local)
```bash
uvicorn tgefd.server:app --reload
```

## Run (Kubernetes, Helm)
```bash
helm upgrade --install tgefd helm/tgefd \
  --set image.repository=YOUR_IMAGE \
  --set image.tag=YOUR_TAG
```

## Debug checklist
- Check `/healthz` and `/readyz`.
- Inspect logs: `kubectl logs deployment/tgefd --tail=200`.
- Validate `/metrics` for Prometheus scrape.
- Verify artifact store credentials.

## Useful envs
- `TGEFD_API_KEY`, `TGEFD_AUTH_MODE`
- `TGEFD_RATE_LIMIT_PER_MINUTE`, `TGEFD_BURST_LIMIT`
- `TGEFD_OTEL_SERVICE_NAME`
- `TGEFD_AUDIT_LOG_PATH`
