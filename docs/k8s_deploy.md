# Test Kubernetes Deploy (minimum)

This folder provides a minimal deployment for test clusters. It includes
liveness/readiness probes and resource limits. Helm is the preferred path;
raw manifests are kept for quick testing.

## Files
- `k8s/tgefd-configmap.yaml`: non-secret env settings
- `k8s/tgefd-secret.yaml`: API key (replace `CHANGE_ME`)
- `k8s/tgefd-deployment.yaml`: Deployment with probes + resources
- `k8s/tgefd-service.yaml`: ClusterIP Service
- `k8s/kustomization.yaml`: one-shot apply
- `helm/tgefd/`: Helm chart (preferred)

## Quick start (test cluster)
```bash
# Build/push image, then edit the image in k8s/tgefd-deployment.yaml
kubectl apply -k k8s
kubectl rollout status deployment/tgefd
kubectl port-forward service/tgefd 8000:80
curl -s http://127.0.0.1:8000/healthz
```

## Helm (preferred)
```bash
helm install tgefd helm/tgefd \
  --set image.repository=YOUR_IMAGE \
  --set image.tag=YOUR_TAG
helm upgrade --install tgefd helm/tgefd
```

## ExternalSecrets (optional)
Set `.Values.externalSecrets.enabled=true` and configure the SecretStore.

## Notes
- Replace `CHANGE_ME` in the Secret (or use your secret manager).
- For production, back the audit log with persistent storage or log shipping.
- Add Ingress, TLS, and OTel exporter endpoints as needed.
