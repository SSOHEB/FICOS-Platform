---
title: Frontend Netlify Deployment
---

# Frontend Netlify Deployment

The production frontend lives in `ficoss/` and is a Vite React app.

## Netlify Build Settings

Use these settings in Netlify:

| Setting | Value |
|---|---|
| Base directory | `ficoss` |
| Build command | `npm run build` |
| Publish directory | `dist` |

Netlify's Vite documentation lists `npm run build` and `dist` as the normal Vite deployment settings.

## Environment Variables

Set this in Netlify under **Site configuration → Environment variables**:

```bash
VITE_FICOS_API_BASE_URL=https://your-backend-api.example.com
```

For frontend-only SIH demo deployment, either leave this pointed to a deployed backend later or use the current UI fallback states. Do not put secret API keys into frontend `VITE_*` variables because Vite exposes them to the browser bundle.

## Continuous Deployment

Connect the GitHub repository to Netlify. After that, every push to the production branch triggers:

```bash
npm install
npm run build
```

If the build succeeds, Netlify publishes the new `dist/` folder automatically over HTTPS.

## SPA Routing

The current app uses hash routes such as:

```text
/#dashboard
/#vessel-intelligence
/#idle-intelligence
```

Hash routing works on static hosting without special rewrites. A fallback rewrite is still configured in `ficoss/netlify.toml` for safety.
