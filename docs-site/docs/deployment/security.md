---
title: Security Checklist
---

# Security Checklist

## Frontend

- Run `npm audit --audit-level=moderate` inside `ficoss/`.
- Run `npm run build` before deployment.
- Keep `VITE_FICOS_API_BASE_URL` pointed to an HTTPS backend in production.
- Do not expose secret keys in `VITE_*` environment variables.
- Use Netlify HTTPS, which is enabled by default for Netlify domains and custom domains after DNS setup.

## Netlify Headers

The frontend includes security headers in `ficoss/netlify.toml`:

- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy` restrictions
- baseline Content Security Policy

## Backend Later

When the backend is deployed, configure:

- strict CORS for the Netlify frontend origin only,
- HTTPS API URL,
- no debug mode,
- no secrets committed to Git,
- rate limiting if the API becomes public.
