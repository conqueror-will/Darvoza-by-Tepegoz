# Burgut — Internal Operations & Situational-Awareness Platform

A sub-project of Darvoza. Burgut ("eagle") is an **internal operations command centre** for a
defence organization's *own* forces, sites, assets and sensors. It collects telemetry and
reports, shows one common picture, runs simple analytics, and drives a human-in-the-loop task
workflow — the same shape as any large organization's network/operations centre.

```
COLLECT -> MONITOR -> ANALYZE -> ACT -> (feedback)
own          one          rules,      human-approved
sensors &    common       alerts,     tasks assigned to
reports      dashboard    dedup       own staff, audited
```

## Scope (read this first)

**In scope**
- Monitoring of the organization's *own* assets, sites, sensors and infrastructure.
- Health/status telemetry: power, fuel, temperature, connectivity, gate/door status of own sites.
- Position reports from the organization's *own* reporting assets (not third parties).
- Staff-authored structured reports and incidents.
- Rule-based alerting (offline detection, threshold breach) and de-duplication.
- A task/ticket workflow: acknowledge → assign to own staff → track to closure.
- Full audit log and per-role access control.

**Explicitly out of scope (hard non-goals)**
- No tracking, profiling or recognition of people, except the existing Darvoza *base-access
  enrollment* of the organization's own personnel, behind an access-policy layer.
- No adversary tracking, "target folders", or fusion of feeds into target pictures.
- No autonomous action of any kind. The platform only recommends and routes work to people.
- No weapons, fire-control, or engagement interfaces, and no integration that would feed one.
- Not a domestic mass-surveillance tool.

If a future request needs any of the above, it is a different project and not built here.

## Common data model

All adapters normalize to these objects (see `backend/server.py` schema):

| Object   | Meaning                                            | Key fields |
|----------|----------------------------------------------------|------------|
| `site`   | A base, depot, post or facility you operate         | id, name, kind, lat, lon |
| `asset`  | An owned thing at a site (generator, vehicle, radio) | id, site_id, name, kind, status |
| `sensor` | A device reporting telemetry about a site/asset     | id, site_id, asset_id, metric, min_ok, max_ok, offline_after_s |
| `reading`| One telemetry sample                                | id, sensor_id, value, ts |
| `report` | A staff-authored structured report                  | id, site_id, author, category, severity, text, ts |
| `alert`  | A condition needing attention (from a rule)         | id, kind, severity, subject, detail, state, ts |
| `task`   | Human work item created to resolve an alert/report  | id, title, alert_id, assignee, priority, state, ts |
| `audit`  | Append-only record of every state change            | id, actor, action, target, ts, detail |

## Architecture

- **Ingest API** — adapters POST readings/reports; everything is normalized and stored.
- **Store** — SQLite (one file, auditable, local). Swappable later for Postgres.
- **Rule engine** — runs on ingest and on a timer: offline detection, threshold breach,
  duplicate-alert suppression.
- **Dashboard** — single-page web client: map, status tiles, alert feed, task board.
- **Workflow** — alerts and reports become tasks; tasks move acknowledge→assigned→in_progress→done.
- **Audit** — every mutating action writes an immutable audit row.

Darvoza fits in as a *base-access* adapter only: its existing camera/face code governs entry of
enrolled own-personnel at your sites, and emits access events to Burgut as ordinary alerts.

## Phases

- **Phase 0 (this scaffold):** data model, ingest, offline + threshold rules, alert→task
  workflow, audit, dashboard, seed data. Stdlib only, runs locally.
- **Phase 1:** role-based access control, report forms, map polish, alert de-dup tuning, tests.
- **Phase 2:** edge node that buffers telemetry offline and syncs; pluggable adapter SDK.
- **Phase 3:** richer analytics (trends, maintenance forecasting), reporting/exports.

Security, classification labelling and access control are present from Phase 0 as stubs and
hardened in Phase 1 — never bolted on at the end.
