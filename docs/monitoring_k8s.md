# Kubernetes Monitoring Integration (Prometheus + Grafana)

This guide connects the existing `/metrics` endpoint to Prometheus and imports
Grafana dashboards. It assumes Prometheus Operator / kube-prometheus-stack.

## 0) Prereqs (CRDs installed)
Check that Prometheus Operator CRDs are present:
```bash
kubectl get crd | rg 'servicemonitors|prometheusrules'
```
If missing, install kube-prometheus-stack or Prometheus Operator before applying
the manifests below.

## 1) Prometheus scrape (ServiceMonitor)
Apply:
```bash
kubectl apply -f k8s/monitoring/servicemonitor.yaml
```

## 2) Alert rules (PrometheusRule)
Apply:
```bash
kubectl apply -f k8s/monitoring/prometheusrule.yaml
```
Note: worker/queue alerts expect metrics `tgefd_job_duration_seconds_bucket`
and `tgefd_queue_depth`. If your async backend exports different names, add
recording rules or update the PrometheusRule accordingly.

## 3) Grafana dashboard import
Apply (Grafana sidecar watches configmaps with `grafana_dashboard=1`):
```bash
kubectl apply -f k8s/monitoring/grafana_dashboard_configmap.yaml
```

## 4) Helm alternative (preferred)
Enable in Helm values:
```yaml
monitoring:
  serviceMonitor:
    enabled: true
  prometheusRule:
    enabled: true
  grafanaDashboard:
    enabled: true
```

## 5) Verify “connected/visible”
Prometheus:
- Check target is UP:
  - UI: `/targets` should show the TGEFD ServiceMonitor target as UP.
  - CLI: `kubectl -n <prom-namespace> port-forward svc/<prom-svc> 9090:9090`
- Run a query and confirm series are present:
  - `tgefd_http_requests_total`
  - `tgefd_http_request_duration_seconds_bucket`

Grafana:
- Confirm dashboard configmap is discovered by sidecar.
- Open the dashboard and verify panels show data.

Alerts:
- In Prometheus “Alerts” page, verify `tgefd-alerts` rules are loaded.
- Use a short-lived load test to trigger `TGEFDErrorRateHigh` or `TGEFDDiscoverEvaluateP95High`.

## 6) Health checks
- `/healthz` and `/readyz` are available for probes.
