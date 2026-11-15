// Burgut dashboard — talks to the local backend JSON API.
const $ = (sel) => document.querySelector(sel);
const api = {
  get: (p) => fetch(p).then((r) => r.json()),
  post: (p, b) => fetch(p, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(b || {}),
  }).then((r) => r.json()),
};

const sevClass = (s) => (s === "critical" ? "crit" : s === "warning" ? "warn" : "ok");
const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString();

// Uzbekistan-ish bounding box for the toy map projection.
const BOX = { minLat: 37.0, maxLat: 45.6, minLon: 56.0, maxLon: 73.2 };
function project(lat, lon) {
  const x = ((lon - BOX.minLon) / (BOX.maxLon - BOX.minLon)) * 100;
  const y = ((BOX.maxLat - lat) / (BOX.maxLat - BOX.minLat)) * 100;
  return { x: Math.max(3, Math.min(97, x)), y: Math.max(8, Math.min(92, y)) };
}

async function load() {
  const [summary, sites, sensors, alerts, tasks] = await Promise.all([
    api.get("/api/summary"), api.get("/api/sites"), api.get("/api/sensors"),
    api.get("/api/alerts"), api.get("/api/tasks"),
  ]);
  renderTiles(summary);
  renderMap(sites, sensors, alerts);
  renderSensors(sensors, sites);
  renderAlerts(alerts);
  renderTasks(tasks);
  $("#status").textContent = "Updated " + new Date().toLocaleTimeString();
}

function renderTiles(s) {
  const tiles = [
    ["Sites", s.sites, ""], ["Assets", s.assets, ""], ["Sensors", s.sensors, ""],
    ["Open alerts", s.alerts_open, s.alerts_open ? "alert" : ""],
    ["Open tasks", s.tasks_open, ""],
  ];
  $("#tiles").innerHTML = tiles.map(([l, n, cls]) =>
    `<div class="tile ${cls}"><div class="n">${n}</div><div class="l">${l}</div></div>`
  ).join("");
}

function renderMap(sites, sensors, alerts) {
  const openBySite = {};
  const sensorSite = Object.fromEntries(sensors.map((s) => [s.id, s.site_id]));
  alerts.filter((a) => a.state === "open").forEach((a) => {
    // subject often contains a sensor id; best-effort attribution to a site
    const sid = Object.keys(sensorSite).find((k) => (a.subject || "").includes(k));
    if (sid) openBySite[sensorSite[sid]] = (openBySite[sensorSite[sid]] || 0) + 1;
  });
  $("#map").innerHTML = sites.map((s) => {
    const { x, y } = project(s.lat, s.lon);
    const bad = openBySite[s.id];
    const color = bad ? "var(--crit)" : "var(--ok)";
    return `<div class="pin" style="left:${x}%;top:${y}%">
      <div class="dot" style="background:${color}"></div>
      <div class="nm">${s.name}${bad ? " ⚠" : ""}</div></div>`;
  }).join("");
}

function renderSensors(sensors, sites) {
  const names = Object.fromEntries(sites.map((s) => [s.id, s.name]));
  const now = Date.now() / 1000;
  $("#sensors tbody").innerHTML = sensors.map((s) => {
    const age = s.last_ts ? now - s.last_ts : null;
    const offline = age !== null && age > (s.offline_after_s || 300);
    const outOfRange = s.last_value !== null &&
      ((s.min_ok !== null && s.last_value < s.min_ok) ||
       (s.max_ok !== null && s.last_value > s.max_ok));
    const st = offline ? ["crit", "offline"] : outOfRange ? ["warn", "out of range"]
      : s.last_ts ? ["ok", "ok"] : ["", "no data"];
    const val = s.last_value === null ? "—"
      : `${s.last_value}${s.unit ? " " + s.unit : ""}`;
    return `<tr><td>${s.id}</td><td>${names[s.site_id] || s.site_id || "—"}</td>
      <td>${s.metric}</td><td>${val}</td>
      <td><span class="badge ${st[0]}">${st[1]}</span></td></tr>`;
  }).join("");
}

function renderAlerts(alerts) {
  const open = alerts.filter((a) => a.state === "open" || a.state === "acknowledged");
  if (!open.length) { $("#alerts").innerHTML = `<div class="muted">No open alerts.</div>`; return; }
  $("#alerts").innerHTML = open.map((a) => `
    <div class="card ${a.severity}">
      <div class="top"><span class="subj">${a.subject}</span>
        <span class="badge ${sevClass(a.severity)}">${a.severity}</span></div>
      <div class="det">${a.detail || ""} · ${fmtTime(a.ts)} · ${a.state}</div>
      <div class="acts">
        ${a.state === "open"
          ? `<button class="ghost" onclick="ackAlert(${a.id})">Acknowledge</button>` : ""}
        <button onclick="taskFromAlert(${a.id}, ${JSON.stringify(a.subject)})">
          Create task</button>
      </div>
    </div>`).join("");
}

function renderTasks(tasks) {
  const live = tasks.filter((t) => t.state !== "done" && t.state !== "cancelled");
  if (!live.length) { $("#tasks").innerHTML = `<div class="muted">No open tasks.</div>`; return; }
  const next = { open: "assigned", assigned: "in_progress", in_progress: "done" };
  $("#tasks").innerHTML = live.map((t) => `
    <div class="card">
      <div class="top"><span class="subj">${t.title}</span>
        <span class="badge ok">${t.state}</span></div>
      <div class="det">assignee: ${t.assignee || "unassigned"} · ${t.priority} ·
        ${fmtTime(t.ts)}</div>
      <div class="acts">
        ${next[t.state]
          ? `<button class="ghost" onclick="advance(${t.id},'${next[t.state]}')">
               → ${next[t.state]}</button>` : ""}
        <button class="ghost" onclick="advance(${t.id},'cancelled')">Cancel</button>
      </div>
    </div>`).join("");
}

// --- actions (prompt for the operator's name so every change is attributable) ---
function who() { return localStorage.getItem("burgut_user") || promptUser(); }
function promptUser() {
  const u = prompt("Your name / callsign (recorded in the audit log):", "operator");
  if (u) localStorage.setItem("burgut_user", u);
  return u || "operator";
}
async function ackAlert(id) { await api.post(`/api/alerts/${id}/ack`, { user: who() }); load(); }
async function taskFromAlert(id, subject) {
  const assignee = prompt("Assign task to:", "duty-team");
  if (assignee === null) return;
  await api.post("/api/tasks",
    { title: "Resolve: " + subject, alert_id: id, assignee, user: who() });
  load();
}
async function advance(id, state) {
  await api.post(`/api/tasks/${id}/status`, { state, user: who() });
  load();
}

$("#refresh").addEventListener("click", load);
load();
setInterval(load, 15000);
