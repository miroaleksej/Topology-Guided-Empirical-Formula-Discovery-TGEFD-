# Production Runbook (Kubernetes)

## Deploy (Helm)
Checklist:
- Namespace exists and context is correct.
- Image is pushed and tag is immutable.
- Secrets/ExternalSecrets are applied.
- Ingress/TLS configured.
- Monitoring enabled if required.

Create namespace (if needed):
```bash
kubectl create namespace tgefd
```

Dry-run (optional):
```bash
kubectl config current-context
kubectl get ns tgefd

cp helm/tgefd/values-prod.yaml.example helm/tgefd/values-prod.yaml
${EDITOR:-vi} helm/tgefd/values-prod.yaml

helm upgrade --install tgefd helm/tgefd \
  --namespace tgefd \
  --values helm/tgefd/values-prod.yaml \
  --dry-run
```

Apply:
```bash
helm upgrade --install tgefd helm/tgefd \
  --namespace tgefd \
  --values helm/tgefd/values-prod.yaml
```

Verify rollout:
```bash
kubectl -n tgefd rollout status deployment/tgefd
kubectl -n tgefd get pods -l app=tgefd
```

## Smoke test
```bash
kubectl -n tgefd port-forward service/tgefd 8000:80
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/readyz
curl -s http://127.0.0.1:8000/metrics | head -n 5
```

## Rollback
```bash
helm -n tgefd history tgefd
helm -n tgefd rollback tgefd <REVISION> --wait --timeout 5m
kubectl -n tgefd rollout status deployment/tgefd
```

## Logs
```bash
kubectl -n tgefd logs deployment/tgefd --tail=200
```

## Cluster command checklist (prod)
```bash
# 1) Ensure secrets / ExternalSecrets are in place
kubectl -n tgefd get externalsecret
kubectl -n tgefd get secret

# 2) Apply monitoring (if not enabled via Helm)
kubectl apply -f k8s/monitoring/servicemonitor.yaml
kubectl apply -f k8s/monitoring/prometheusrule.yaml
kubectl apply -f k8s/monitoring/grafana_dashboard_configmap.yaml

# 3) Apply ingress (if not enabled via Helm)
kubectl apply -f k8s/ingress/nginx_ingress.yaml

# 4) Deploy or upgrade
helm upgrade --install tgefd helm/tgefd \
  --namespace tgefd \
  --values helm/tgefd/values-prod.yaml

# 5) Smoke test
kubectl -n tgefd port-forward service/tgefd 8000:80
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/readyz
curl -s http://127.0.0.1:8000/metrics | head -n 5
```
