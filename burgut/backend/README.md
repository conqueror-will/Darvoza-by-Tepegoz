# Burgut backend

Dependency-free (Python stdlib only) demo backend for the Burgut operations-picture concept.
It is a **layout / proof-of-concept**, not a production system — see [`../docs/PLAN.md`](../docs/PLAN.md)
for scope and the hard non-goals (no people-tracking, no targeting, no autonomous action).

## Run

```bash
python3 server.py --seed      # create demo data (3 sites, 4 assets, 5 sensors) and serve
python3 server.py             # serve existing db
# then open http://127.0.0.1:8787  in a browser
```

Options: `--port 8787`, `--host 127.0.0.1`, `--db burgut.db`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/ingest/reading` | `{sensor_id, value, ts?}` — telemetry sample |
| POST | `/api/ingest/report`  | `{site_id, author, category, severity, text}` — staff report |
| GET  | `/api/summary`        | counts for the dashboard tiles |
| GET  | `/api/sites` `/api/assets` `/api/sensors` | inventory |
| GET  | `/api/alerts` `/api/reports` `/api/tasks` `/api/audit` | lists |
| POST | `/api/alerts/{id}/ack` | `{user}` — acknowledge |
| POST | `/api/tasks`          | `{title, alert_id?, assignee, priority, user}` — create work item |
| POST | `/api/tasks/{id}/status` | `{state, user}` — open→assigned→in_progress→done/cancelled |

## Rules

- **Threshold** — a reading outside `[min_ok, max_ok]` raises a `warning`; a later in-range
  reading auto-resolves it.
- **Offline** — a watchdog thread raises a `critical` if a sensor goes silent past
  `offline_after_s`; recovery auto-resolves it.
- **De-dup** — only one open alert per condition (unique `dedup_key`).

Every mutating action writes an append-only `audit` row (`actor, action, target, ts, detail`).
