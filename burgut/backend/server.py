"""
Burgut backend - internal operations & situational-awareness platform.

Single-file, dependency-free (Python standard library only): http.server + sqlite3.
Serves a JSON API and the static web dashboard from one process.

    python3 server.py                 # serves on http://127.0.0.1:8787
    python3 server.py --seed          # (re)create demo data, then serve
    python3 server.py --port 9000     # choose a port
    python3 server.py --db burgut.db  # choose the SQLite file

SCOPE: monitors the organization's OWN sites, assets and sensors. It does NOT track people,
build target pictures, or take any autonomous action. See ../docs/PLAN.md.
"""

import os
import sys
import json
import time
import sqlite3
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(os.path.dirname(HERE), "web")
DEFAULT_DB = os.path.join(HERE, "burgut.db")

# ------------------------------------------------------------------ database

SCHEMA = """
CREATE TABLE IF NOT EXISTS site (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT, lat REAL, lon REAL
);
CREATE TABLE IF NOT EXISTS asset (
    id TEXT PRIMARY KEY, site_id TEXT, name TEXT NOT NULL, kind TEXT,
    status TEXT DEFAULT 'ok'
);
CREATE TABLE IF NOT EXISTS sensor (
    id TEXT PRIMARY KEY, site_id TEXT, asset_id TEXT, metric TEXT NOT NULL,
    unit TEXT, min_ok REAL, max_ok REAL, offline_after_s INTEGER DEFAULT 300,
    last_value REAL, last_ts REAL
);
CREATE TABLE IF NOT EXISTS reading (
    id INTEGER PRIMARY KEY AUTOINCREMENT, sensor_id TEXT NOT NULL,
    value REAL NOT NULL, ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS report (
    id INTEGER PRIMARY KEY AUTOINCREMENT, site_id TEXT, author TEXT,
    category TEXT, severity TEXT, text TEXT, ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS alert (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, severity TEXT,
    subject TEXT, detail TEXT, state TEXT DEFAULT 'open',
    dedup_key TEXT, ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS task (
    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, alert_id INTEGER,
    assignee TEXT, priority TEXT DEFAULT 'normal', state TEXT DEFAULT 'open',
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT, action TEXT,
    target TEXT, detail TEXT, ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reading_sensor ON reading(sensor_id, ts);
CREATE INDEX IF NOT EXISTS idx_alert_state ON alert(state);
CREATE UNIQUE INDEX IF NOT EXISTS idx_alert_dedup ON alert(dedup_key) WHERE state='open';
"""

_db_lock = threading.Lock()


def connect(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn):
    with _db_lock:
        conn.executescript(SCHEMA)
        conn.commit()


def audit(conn, actor, action, target, detail=""):
    conn.execute(
        "INSERT INTO audit(actor, action, target, detail, ts) VALUES (?,?,?,?,?)",
        (actor, action, target, detail, time.time()),
    )


# ------------------------------------------------------------------ rule engine

def raise_alert(conn, kind, severity, subject, detail, dedup_key):
    """Create an alert unless an open one with the same dedup_key already exists."""
    cur = conn.execute(
        "SELECT id FROM alert WHERE dedup_key=? AND state='open'", (dedup_key,)
    )
    if cur.fetchone():
        return None  # suppressed duplicate
    cur = conn.execute(
        "INSERT INTO alert(kind, severity, subject, detail, state, dedup_key, ts)"
        " VALUES (?,?,?,?, 'open', ?, ?)",
        (kind, severity, subject, detail, dedup_key, time.time()),
    )
    audit(conn, "system", "alert.raise", f"alert:{cur.lastrowid}", f"{kind} {subject}")
    return cur.lastrowid


def evaluate_reading(conn, sensor, value, ts):
    """Threshold rule: fire if a reading is outside [min_ok, max_ok]."""
    sid = sensor["id"]
    lo, hi = sensor["min_ok"], sensor["max_ok"]
    breach = (lo is not None and value < lo) or (hi is not None and value > hi)
    if breach:
        raise_alert(
            conn, "threshold", "warning",
            f"{sensor['metric']} out of range on {sid}",
            f"value={value} ok=[{lo},{hi}] unit={sensor['unit'] or ''}",
            dedup_key=f"threshold:{sid}",
        )
    else:
        # value back in range -> auto-resolve a matching open threshold alert
        conn.execute(
            "UPDATE alert SET state='resolved' WHERE dedup_key=? AND state='open'",
            (f"threshold:{sid}",),
        )


def check_offline(conn):
    """Offline rule: fire if a sensor has not reported within offline_after_s."""
    now = time.time()
    rows = conn.execute("SELECT * FROM sensor").fetchall()
    for s in rows:
        if s["last_ts"] is None:
            continue
        if now - s["last_ts"] > (s["offline_after_s"] or 300):
            raise_alert(
                conn, "offline", "critical",
                f"sensor {s['id']} offline",
                f"no reading for {int(now - s['last_ts'])}s (metric {s['metric']})",
                dedup_key=f"offline:{s['id']}",
            )
        else:
            conn.execute(
                "UPDATE alert SET state='resolved' WHERE dedup_key=? AND state='open'",
                (f"offline:{s['id']}",),
            )
    conn.commit()


def offline_watchdog(conn, stop_event, interval=30):
    while not stop_event.wait(interval):
        try:
            with _db_lock:
                check_offline(conn)
        except Exception as exc:  # never let the watchdog die silently
            sys.stderr.write(f"[watchdog] {exc}\n")


# ------------------------------------------------------------------ API handlers

def rows(conn, sql, args=()):
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


class Api:
    """Maps (method, path) -> handler. Each returns a JSON-serializable object."""

    def __init__(self, conn):
        self.conn = conn

    # ---- collect ----
    def ingest_reading(self, body):
        sid = body["sensor_id"]
        value = float(body["value"])
        ts = float(body.get("ts") or time.time())
        with _db_lock:
            sensor = self.conn.execute(
                "SELECT * FROM sensor WHERE id=?", (sid,)
            ).fetchone()
            if not sensor:
                return {"error": f"unknown sensor {sid}"}, 404
            self.conn.execute(
                "INSERT INTO reading(sensor_id, value, ts) VALUES (?,?,?)",
                (sid, value, ts),
            )
            self.conn.execute(
                "UPDATE sensor SET last_value=?, last_ts=? WHERE id=?", (value, ts, sid)
            )
            evaluate_reading(self.conn, sensor, value, ts)
            self.conn.commit()
        return {"ok": True}, 200

    def ingest_report(self, body):
        with _db_lock:
            cur = self.conn.execute(
                "INSERT INTO report(site_id, author, category, severity, text, ts)"
                " VALUES (?,?,?,?,?,?)",
                (body.get("site_id"), body.get("author"), body.get("category"),
                 body.get("severity", "info"), body.get("text", ""), time.time()),
            )
            # a report at warning/critical severity also raises an alert
            if body.get("severity") in ("warning", "critical"):
                raise_alert(
                    self.conn, "report", body["severity"],
                    f"report: {body.get('category','')}",
                    (body.get("text", "")[:200]),
                    dedup_key=f"report:{cur.lastrowid}",
                )
            audit(self.conn, body.get("author", "unknown"), "report.create",
                  f"report:{cur.lastrowid}", body.get("category", ""))
            self.conn.commit()
        return {"ok": True, "id": cur.lastrowid}, 200

    # ---- monitor ----
    def summary(self, _):
        c = self.conn
        return {
            "sites": c.execute("SELECT COUNT(*) FROM site").fetchone()[0],
            "assets": c.execute("SELECT COUNT(*) FROM asset").fetchone()[0],
            "sensors": c.execute("SELECT COUNT(*) FROM sensor").fetchone()[0],
            "alerts_open": c.execute(
                "SELECT COUNT(*) FROM alert WHERE state='open'").fetchone()[0],
            "tasks_open": c.execute(
                "SELECT COUNT(*) FROM task WHERE state NOT IN ('done','cancelled')"
            ).fetchone()[0],
        }, 200

    def list_sites(self, _):
        return rows(self.conn, "SELECT * FROM site ORDER BY name"), 200

    def list_assets(self, _):
        return rows(self.conn, "SELECT * FROM asset ORDER BY name"), 200

    def list_sensors(self, _):
        return rows(self.conn,
                    "SELECT * FROM sensor ORDER BY site_id, metric"), 200

    def list_alerts(self, _):
        return rows(self.conn,
                    "SELECT * FROM alert ORDER BY ts DESC LIMIT 200"), 200

    def list_reports(self, _):
        return rows(self.conn,
                    "SELECT * FROM report ORDER BY ts DESC LIMIT 200"), 200

    def list_tasks(self, _):
        return rows(self.conn, "SELECT * FROM task ORDER BY ts DESC LIMIT 200"), 200

    def list_audit(self, _):
        return rows(self.conn, "SELECT * FROM audit ORDER BY ts DESC LIMIT 200"), 200

    # ---- act (human in the loop) ----
    def ack_alert(self, body, alert_id):
        actor = body.get("user", "unknown")
        with _db_lock:
            self.conn.execute(
                "UPDATE alert SET state='acknowledged' WHERE id=? AND state='open'",
                (alert_id,),
            )
            audit(self.conn, actor, "alert.ack", f"alert:{alert_id}")
            self.conn.commit()
        return {"ok": True}, 200

    def create_task(self, body):
        actor = body.get("user", "unknown")
        with _db_lock:
            cur = self.conn.execute(
                "INSERT INTO task(title, alert_id, assignee, priority, state, ts)"
                " VALUES (?,?,?,?, 'open', ?)",
                (body["title"], body.get("alert_id"), body.get("assignee"),
                 body.get("priority", "normal"), time.time()),
            )
            audit(self.conn, actor, "task.create", f"task:{cur.lastrowid}",
                  body.get("title", ""))
            self.conn.commit()
        return {"ok": True, "id": cur.lastrowid}, 200

    def task_status(self, body, task_id):
        actor = body.get("user", "unknown")
        state = body["state"]
        allowed = {"open", "assigned", "in_progress", "done", "cancelled"}
        if state not in allowed:
            return {"error": f"bad state {state}"}, 400
        with _db_lock:
            self.conn.execute("UPDATE task SET state=? WHERE id=?", (state, task_id))
            audit(self.conn, actor, "task.status", f"task:{task_id}", state)
            self.conn.commit()
        return {"ok": True}, 200


# ------------------------------------------------------------------ http layer

def make_handler(api):
    routes_get = {
        "/api/summary": api.summary,
        "/api/sites": api.list_sites,
        "/api/assets": api.list_assets,
        "/api/sensors": api.list_sensors,
        "/api/alerts": api.list_alerts,
        "/api/reports": api.list_reports,
        "/api/tasks": api.list_tasks,
        "/api/audit": api.list_audit,
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "Burgut/0.1"

        def log_message(self, fmt, *args):  # quieter default logging
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _json(self, obj, status=200):
            payload = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _static(self, path):
            if path == "/" or path == "":
                path = "/index.html"
            safe = os.path.normpath(path).lstrip("/")
            full = os.path.join(WEB_DIR, safe)
            if not full.startswith(WEB_DIR) or not os.path.isfile(full):
                self.send_error(404)
                return
            ctype = {
                ".html": "text/html", ".js": "application/javascript",
                ".css": "text/css", ".json": "application/json",
            }.get(os.path.splitext(full)[1], "application/octet-stream")
            data = open(full, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            return json.loads(self.rfile.read(length).decode() or "{}")

        def do_GET(self):
            parsed = urlparse(self.path)
            fn = routes_get.get(parsed.path)
            if fn:
                try:
                    result = fn(parse_qs(parsed.query))
                    body, status = result if isinstance(result, tuple) else (result, 200)
                    self._json(body, status)
                except Exception as exc:
                    self._json({"error": str(exc)}, 500)
                return
            self._static(parsed.path)

        def do_POST(self):
            parsed = urlparse(self.path)
            p = parsed.path
            try:
                body = self._body()
                if p == "/api/ingest/reading":
                    out, st = api.ingest_reading(body)
                elif p == "/api/ingest/report":
                    out, st = api.ingest_report(body)
                elif p == "/api/tasks":
                    out, st = api.create_task(body)
                elif p.startswith("/api/alerts/") and p.endswith("/ack"):
                    out, st = api.ack_alert(body, int(p.split("/")[3]))
                elif p.startswith("/api/tasks/") and p.endswith("/status"):
                    out, st = api.task_status(body, int(p.split("/")[3]))
                else:
                    out, st = {"error": "not found"}, 404
                self._json(out, st)
            except KeyError as exc:
                self._json({"error": f"missing field {exc}"}, 400)
            except Exception as exc:
                self._json({"error": str(exc)}, 500)

    return Handler


# ------------------------------------------------------------------ seed data

def seed(conn):
    with _db_lock:
        for t in ("audit", "task", "alert", "report", "reading",
                  "sensor", "asset", "site"):
            conn.execute(f"DELETE FROM {t}")
        sites = [
            ("site-tash", "HQ Depot Tashkent", "depot", 41.311, 69.279),
            ("site-samr", "Depot Samarkand", "depot", 39.654, 66.975),
            ("site-nukus", "Post Nukus", "post", 42.461, 59.610),
        ]
        conn.executemany(
            "INSERT INTO site(id,name,kind,lat,lon) VALUES (?,?,?,?,?)", sites)
        assets = [
            ("gen-1", "site-tash", "Generator A", "generator", "ok"),
            ("fuel-1", "site-tash", "Fuel Tank 1", "fuel_tank", "ok"),
            ("gen-2", "site-samr", "Generator B", "generator", "ok"),
            ("veh-1", "site-nukus", "Support Truck 1", "vehicle", "ok"),
        ]
        conn.executemany(
            "INSERT INTO asset(id,site_id,name,kind,status) VALUES (?,?,?,?,?)", assets)
        now = time.time()
        sensors = [
            ("s-gen1-temp", "site-tash", "gen-1", "temperature", "C", 10, 90, 300),
            ("s-fuel1-lvl", "site-tash", "fuel-1", "fuel_level", "%", 20, 100, 600),
            ("s-tash-power", "site-tash", None, "grid_power", "V", 210, 245, 120),
            ("s-samr-gate", "site-samr", None, "gate_status", "", 0, 1, 300),
            ("s-nukus-link", "site-nukus", None, "uplink", "ms", 0, 400, 180),
        ]
        for sid, site, asset, metric, unit, lo, hi, off in sensors:
            conn.execute(
                "INSERT INTO sensor(id,site_id,asset_id,metric,unit,min_ok,max_ok,"
                "offline_after_s,last_value,last_ts) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (sid, site, asset, metric, unit, lo, hi, off, None, None),
            )
        # a few healthy readings + one threshold breach to populate the board
        demo = [
            ("s-gen1-temp", 62, now - 10),
            ("s-fuel1-lvl", 74, now - 20),
            ("s-tash-power", 231, now - 5),
            ("s-samr-gate", 0, now - 30),
            ("s-nukus-link", 520, now - 15),   # breach: uplink latency too high
        ]
        for sid, val, ts in demo:
            conn.execute(
                "INSERT INTO reading(sensor_id,value,ts) VALUES (?,?,?)", (sid, val, ts))
            conn.execute(
                "UPDATE sensor SET last_value=?, last_ts=? WHERE id=?", (val, ts, sid))
            s = conn.execute("SELECT * FROM sensor WHERE id=?", (sid,)).fetchone()
            evaluate_reading(conn, s, val, ts)
        audit(conn, "system", "seed", "db", "demo data loaded")
        conn.commit()
    print("Seeded demo data: 3 sites, 4 assets, 5 sensors.")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Burgut backend")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--seed", action="store_true", help="reset + load demo data")
    args = ap.parse_args()

    conn = connect(args.db)
    init_db(conn)
    if args.seed:
        seed(conn)

    api = Api(conn)
    stop = threading.Event()
    wd = threading.Thread(target=offline_watchdog, args=(conn, stop), daemon=True)
    wd.start()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(api))
    print(f"Burgut backend on http://{args.host}:{args.port}  (db: {args.db})")
    print("Open that URL in a browser for the dashboard. Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.server_close()


if __name__ == "__main__":
    main()
