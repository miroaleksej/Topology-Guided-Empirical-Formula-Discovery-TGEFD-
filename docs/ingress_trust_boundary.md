# Ingress Trust Boundary (Identity Headers)

## Goal
Ensure identity headers and `X-Forwarded-For` are only trusted when set by a
known ingress/proxy.

## Recommendations
- Terminate TLS at the ingress and **overwrite** forwarded headers.
- Configure ingress to set a dedicated identity header (e.g., `x-auth-request-user`).
- In TGEFD, set `TGEFD_IDENTITY_HEADER=x-auth-request-user`.
- Do not forward user-supplied identity headers from the public edge.

## Step-by-step (NGINX Ingress)
1) Create/ensure TLS secret:
```bash
kubectl create secret tls tgefd-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key
```

2) Apply the NGINX ingress template:
```bash
kubectl apply -f k8s/ingress/nginx_ingress.yaml
```
Note: `configuration-snippet` requires `nginx.ingress.kubernetes.io/enable-snippets=true`
in the ingress controller. If snippets are disabled, use the `proxy-set-headers`
ConfigMap template embedded in the same file.

3) Configure identity header in the TGEFD deployment:
- Set `TGEFD_IDENTITY_HEADER=x-auth-request-user`
- If using JWT or mTLS, also set `TGEFD_AUTH_MODE=jwt|mtls` and related env vars.

4) (Optional) Wire external auth to populate identity header:
- Use oauth2-proxy or another auth service.
- In `k8s/ingress/nginx_ingress.yaml`, uncomment:
  - `nginx.ingress.kubernetes.io/auth-url`
  - `nginx.ingress.kubernetes.io/auth-response-headers`

5) Verify header trust boundary:
- Send a request with a forged `x-auth-request-user` and confirm it is **not**
  accepted unless set by the ingress/auth layer.

## Step-by-step (Envoy Gateway / Gateway API)
1) Create/ensure TLS secret (same as above).
2) Apply the Envoy Gateway template:
```bash
kubectl apply -f k8s/ingress/envoy_gateway.yaml
```
3) Wire external auth (ext_authz/JWT/mTLS) at the gateway level so that the
identity header is injected by Envoy, not by the client.
4) Set `TGEFD_IDENTITY_HEADER` to the injected header name.

## Trust-boundary checks
Use one of the following strategies to avoid spoofing:
- **NGINX Ingress:** overwrite `X-Forwarded-*` and identity headers in the ingress
  configuration (see `k8s/ingress/nginx_ingress.yaml`).
- **Envoy Gateway:** strip incoming identity headers at the gateway and only
  inject trusted identity after authn/authz (see `k8s/ingress/envoy_gateway.yaml`).

## mTLS
- If using mTLS at ingress, forward verified subject in `x-ssl-client-subject`
  and set `TGEFD_MTLS_IDENTITY_HEADER=x-ssl-client-subject`.
