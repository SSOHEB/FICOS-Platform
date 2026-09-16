---
title: FICOS Documentation
slug: /
---

# FICOS Documentation

FICOS is a freight intelligence and chartering optimization project that combines:

- freight rate forecasting,
- vessel and port feasibility checks,
- operational risk breakdowns,
- idle and repositioning intelligence,
- policy recommendations for BUY / WAIT / FLEXIBLE decisions.

This documentation site is generated from the project README files, architecture notes, audit reports, model-selection notes, and diagnostic images in the repository.

## Quick Links

- [Frontend Netlify Deployment](deployment/frontend-netlify.md)
- [Docs Netlify Deployment](deployment/docs-netlify.md)
- [Security Checklist](deployment/security.md)
- [Architecture](project-docs/architecture.md)
- [Master Evaluation Report](reports/MASTER_EVALUATION_REPORT.md)

## Current Deployment Strategy

For the SIH presentation, deploy the frontend as a static Netlify site first. The backend can remain local or be deployed later as an API service. Until the backend is deployed, configure the frontend to either:

- call a deployed backend API URL when available, or
- clearly show fallback/unavailable states for backend-driven cards.

The frontend has already been structured to read the backend URL from `VITE_FICOS_API_BASE_URL`.
