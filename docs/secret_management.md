# Secret Management (Vault / ExternalSecrets)

## Recommended approach
Use External Secrets Operator with a Vault or cloud secret backend.
The application expects `TGEFD_API_KEY` (and optionally JWT secrets/keys).

## Helm (preferred)
Enable ExternalSecrets in Helm:
```bash
helm upgrade --install tgefd helm/tgefd \
  --set externalSecrets.enabled=true \
  --set externalSecrets.secretStoreRef.name=vault-backend \
  --set externalSecrets.secretStoreRef.kind=ClusterSecretStore
```

## Raw manifest example
See `k8s/externalsecret.yaml` for a reference ExternalSecret.

## Notes
- Rotate secrets periodically.
- Use least-privilege access on the SecretStore.
- Do not place plaintext secrets in manifests.
