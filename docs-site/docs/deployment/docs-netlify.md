---
title: Documentation Netlify Deployment
---

# Documentation Netlify Deployment

The documentation portal lives in `docs-site/`.

## Build Settings

Use these Netlify settings:

| Setting | Value |
|---|---|
| Base directory | `docs-site` |
| Build command | `npm run build` |
| Publish directory | `build` |

Docusaurus builds static files into `build/`; Netlify can publish that directory directly.

## Source Sync

Before each docs build, this command runs automatically:

```bash
npm run sync
```

It copies the root README files, `docs/*.md`, `reports/*.md`, and diagnostic images into the Docusaurus site so the documentation stays aligned with the project.

## Updating Live Docs

After the repository is connected to Netlify, update any source markdown or image, commit, and push. Netlify will rebuild the documentation site and publish the updated version.
