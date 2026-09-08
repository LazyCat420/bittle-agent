/**
 * Training dashboard: every run, every gate, every GLM cycle — live.
 *
 * Data: /api/training/runs (leaderboard per suite), /api/training/run/{id} (curve + benchmark),
 * /api/training/tasks (catalogue + prerequisites), /api/training/compare, /api/training/benchmark/{id},
 * /api/training/sessions[/live] (GLM's training-mode sessions, recorded server-side so every viewer
 * sees the same cycles). Charts are plain canvas — no dependencies.
 */

const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(d);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const PALETTE = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#a78bfa', '#22d3ee', '#f472b6', '#84cc16', '#fb923c', '#e879f9'];

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (e) { /* ignore */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return r.json();
}

// ── canvas charts ─────────────────────────────────────────────────────────

function prepCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 220;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

/** series: [{label, color, points: [[x, y], ...]}]; opts: {xlabel, ylabel, title, yZero} */
export function lineChart(canvas, series, opts = {}) {
  const { ctx, w, h } = prepCanvas(canvas);
  const pad = { l: 48, r: 12, t: 22, b: 30 };
  const pts = series.flatMap(s => s.points).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  ctx.font = '11px Inter, sans-serif'; ctx.fillStyle = '#8b9bb4';
  if (opts.title) { ctx.fillStyle = '#f0f6fc'; ctx.fillText(opts.title, pad.l, 14); ctx.fillStyle = '#8b9bb4'; }
  if (!pts.length) { ctx.fillText('no data yet', pad.l, h / 2); return; }
  let xmin = Math.min(...pts.map(p => p[0])), xmax = Math.max(...pts.map(p => p[0]));
  let ymin = Math.min(...pts.map(p => p[1])), ymax = Math.max(...pts.map(p => p[1]));
  if (opts.yZero) ymin = Math.min(0, ymin);
  if (xmax === xmin) xmax = xmin + 1;
  if (ymax === ymin) ymax = ymin + 1;
  const py = ymax - ymin; ymin -= py * 0.05; ymax += py * 0.05;
  const X = x => pad.l + (x - xmin) / (xmax - xmin) * (w - pad.l - pad.r);
  const Y = y => h - pad.b - (y - ymin) / (ymax - ymin) * (h - pad.t - pad.b);
  ctx.strokeStyle = '#263042'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = ymin + (ymax - ymin) * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, Y(y)); ctx.lineTo(w - pad.r, Y(y)); ctx.stroke();
    ctx.fillText(fmt(y, Math.abs(ymax) < 5 ? 2 : 0), 4, Y(y) + 4);
  }
  for (let i = 0; i <= 4; i++) {
    const x = xmin + (xmax - xmin) * i / 4;
    ctx.fillText((opts.xfmt || (v => fmt(v, 1)))(x), X(x) - 10, h - 8);
  }
  if (opts.xlabel) ctx.fillText(opts.xlabel, w - pad.r - 60, h - 8);
  series.forEach((s, i) => {
    const p = s.points.filter(q => Number.isFinite(q[0]) && Number.isFinite(q[1]));
    if (!p.length) return;
    ctx.strokeStyle = s.color || PALETTE[i % PALETTE.length]; ctx.lineWidth = 2; ctx.beginPath();
    p.forEach((q, k) => k ? ctx.lineTo(X(q[0]), Y(q[1])) : ctx.moveTo(X(q[0]), Y(q[1])));
    ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    p.forEach(q => { ctx.beginPath(); ctx.arc(X(q[0]), Y(q[1]), 2.5, 0, Math.PI * 2); ctx.fill(); });
  });
  // legend
  let lx = pad.l + 4;
  series.forEach((s, i) => {
    ctx.fillStyle = s.color || PALETTE[i % PALETTE.length]; ctx.fillRect(lx, pad.t - 6, 10, 3);
    ctx.fillStyle = '#8b9bb4'; ctx.fillText(s.label || '', lx + 14, pad.t - 2);
    lx += 14 + ctx.measureText(s.label || '').width + 14;
  });
}

/** Grouped bars of value/threshold per gate (below 1 = pass). groups: [{label, color, ratios: {gate: {ratio, pass}}}] */
export function gateBars(canvas, gates, groups, opts = {}) {
  const { ctx, w, h } = prepCanvas(canvas);
  const pad = { l: 36, r: 8, t: 22, b: 84 };
  ctx.font = '10px Inter, sans-serif'; ctx.fillStyle = '#8b9bb4';
  if (opts.title) { ctx.fillStyle = '#f0f6fc'; ctx.font = '11px Inter, sans-serif'; ctx.fillText(opts.title, pad.l, 14); ctx.font = '10px Inter, sans-serif'; ctx.fillStyle = '#8b9bb4'; }
  if (!gates.length) { ctx.fillText('no gates', pad.l, h / 2); return; }
  const ymax = 2.1;
  const Y = y => h - pad.b - Math.min(y, ymax) / ymax * (h - pad.t - pad.b);
  const slot = (w - pad.l - pad.r) / gates.length;
  const bw = Math.max(3, (slot * 0.8) / groups.length);
  ctx.strokeStyle = '#263042';
  [0.5, 1.0, 1.5, 2.0].forEach(v => { ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke(); ctx.fillText(fmt(v, 1), 6, Y(v) + 3); });
  ctx.strokeStyle = '#f0f6fc'; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(pad.l, Y(1)); ctx.lineTo(w - pad.r, Y(1)); ctx.stroke(); ctx.setLineDash([]);
  gates.forEach((g, gi) => {
    groups.forEach((grp, k) => {
      const r = grp.ratios[g];
      if (!r) return;
      const x = pad.l + gi * slot + slot * 0.1 + k * bw;
      const ratio = Math.min(r.ratio, ymax);
      ctx.fillStyle = r.pass === false ? '#ef4444' : (r.pass === true ? (grp.color || '#10b981') : '#3b475d');
      ctx.globalAlpha = r.pass === false ? 0.95 : 0.85;
      ctx.fillRect(x, Y(ratio), bw - 1, Y(0) - Y(ratio));
      ctx.globalAlpha = 1;
      if (r.ratio > ymax) { ctx.fillStyle = '#f0f6fc'; ctx.fillText('▲', x, Y(ymax) - 2); }
    });
    ctx.save(); ctx.translate(pad.l + gi * slot + slot / 2, h - pad.b + 6); ctx.rotate(Math.PI / 3);
    ctx.fillStyle = '#8b9bb4'; ctx.fillText(g, 0, 0); ctx.restore();
  });
  let lx = pad.l + 4;
  groups.forEach((grp, i) => {
    ctx.fillStyle = grp.color || PALETTE[i % PALETTE.length]; ctx.fillRect(lx, pad.t - 6, 10, 3);
    ctx.fillStyle = '#8b9bb4'; ctx.fillText(grp.label, lx + 14, pad.t - 2);
    lx += 14 + ctx.measureText(grp.label).width + 14;
  });
}

function gateRatio(g) {
  if (g.value === null || g.value === undefined) return null;
  const thr = Array.isArray(g.threshold) ? g.threshold[1] : g.threshold;
  const v = Number(g.value);
  if (g.op === '>=' || g.op === '>') return { ratio: v > 0 ? Number(thr) / v : 2.5, pass: g.pass };
  if (!thr) return { ratio: v <= 0 ? 0 : 2.5, pass: g.pass };
  return { ratio: v / Number(thr), pass: g.pass };
}

// ── the dashboard ─────────────────────────────────────────────────────────

export class TrainingDashboard {
  constructor(opts = {}) {
    this.root = typeof opts.root === 'string' ? document.querySelector(opts.root) : opts.root;
    this.viewer = opts.viewer || null;
    this.suite = null;
    this.runs = [];
    this.bestBySuite = {};
    this.tasks = [];
    this.selected = null;       // run id shown in detail
    this.compare = new Set();   // run ids ticked for compare
    this.session = null;        // session dict shown in the GLM panel
    this.sessions = [];
    this.timer = null;
    this.es = null;
    this.lastDetail = null;
    this.render();
    this.connectLive();
    this.refresh().catch(() => {});
    this.schedule();
    window.addEventListener('resize', () => this.redrawCharts());
  }

  // ── skeleton ───────────────────────────────────────────────────────
  render() {
    if (!this.root) return;
    this.root.innerHTML = `
      <div class="td-bar">
        <span class="td-chip" id="tdTrainer">trainer: …</span>
        <span class="td-chip" id="tdLive">no live GLM session</span>
        <label class="td-chip">suite <select id="tdSuite"></select></label>
        <button class="vbtn td-small" id="tdRefresh">↻ refresh</button>
        <span class="td-muted" id="tdUpdated"></span>
      </div>
      <div class="td-tasks" id="tdTasks"></div>
      <div class="td-grid">
        <section class="td-card td-span2">
          <div class="td-head"><span>🧠 GLM training session</span>
            <select id="tdSessionPick" class="td-small"></select></div>
          <div id="tdSession" class="td-session"></div>
        </section>
        <section class="td-card td-span2">
          <div class="td-head"><span>🏆 Leaderboard <span class="td-muted" id="tdSuiteLabel"></span></span>
            <span class="td-muted">tick runs to compare · click a row for detail</span></div>
          <div class="td-tablewrap"><table class="td-table" id="tdRuns"></table></div>
          <div class="td-muted" id="tdBaselines"></div>
        </section>
        <section class="td-card">
          <div class="td-head"><span>📈 Run detail</span><span id="tdDetailTitle" class="td-muted"></span></div>
          <div id="tdDetail"></div>
        </section>
        <section class="td-card">
          <div class="td-head"><span>⚖️ Compare</span><span class="td-muted" id="tdCompareCount">0 selected</span></div>
          <div id="tdCompare" class="td-muted">Tick two or more runs in the leaderboard.</div>
        </section>
      </div>`;
    this.root.querySelector('#tdRefresh').onclick = () => this.refresh();
    this.root.querySelector('#tdSuite').onchange = (e) => { this.suite = e.target.value; this.refresh(); };
    this.root.querySelector('#tdSessionPick').onchange = (e) => this.loadSession(e.target.value);
  }

  schedule() {
    clearTimeout(this.timer);
    const busy = this.runs.some(r => ['queued', 'training', 'benchmarking'].includes(r.status)) || !!this.liveSessionId;
    this.timer = setTimeout(() => this.refresh().finally(() => this.schedule()), busy ? 5000 : 20000);
  }

  // ── data ───────────────────────────────────────────────────────────
  async refresh() {
    const q = this.suite ? `&suite=${encodeURIComponent(this.suite)}` : '';
    const [runsRes, tasksRes, health] = await Promise.all([
      getJSON(`/api/training/runs?sort=score&limit=60${q}`).catch(e => ({ error: e.message })),
      getJSON('/api/training/tasks').catch(() => null),
      getJSON('/api/training/health').catch(() => ({ ok: false })),
    ]);
    const tr = this.root.querySelector('#tdTrainer');
    tr.textContent = health.ok ? `trainer ok · ${health.health?.physics_impl || ''} · ${health.health?.gpu?.name || ''} · jobs ${(health.health?.jobs?.running || []).length} running / ${(health.health?.jobs?.train_queue || []).length} queued` : `trainer: ${health.error || 'unreachable'}`;
    tr.className = 'td-chip ' + (health.ok ? 'td-ok' : 'td-bad');
    if (runsRes.error) { this.root.querySelector('#tdRuns').innerHTML = `<tr><td class="td-bad">${esc(runsRes.error)}</td></tr>`; return; }
    this.runs = runsRes.runs || [];
    this.bestBySuite = runsRes.best_by_suite || {};
    this.baselines = runsRes.baselines || {};
    this.baselinesSuite = runsRes.baselines_suite;
    if (tasksRes) this.tasks = tasksRes.tasks || [];
    this.renderSuites();
    this.renderTasks();
    this.renderRuns();
    this.root.querySelector('#tdUpdated').textContent = `updated ${new Date().toLocaleTimeString()}`;
    if (this.selected) await this.loadDetail(this.selected, true);
    else if (this.runs.length) await this.loadDetail((this.runs.find(r => r.gates_total) || this.runs[0]).run_id, true);
    if (this.compare.size >= 2) await this.renderCompare();
    if (!this.sessions.length) await this.loadSessions();
  }

  renderSuites() {
    const sel = this.root.querySelector('#tdSuite');
    const suites = new Set([...Object.keys(this.bestBySuite), ...this.tasks.map(t => t.suite), ...this.runs.map(r => r.suite).filter(Boolean)]);
    const cur = this.suite || '';
    sel.innerHTML = `<option value="">all suites</option>` + [...suites].sort().map(s => `<option value="${esc(s)}" ${s === cur ? 'selected' : ''}>${esc(s)}</option>`).join('');
    this.root.querySelector('#tdSuiteLabel').textContent = this.suite ? `· ${this.suite}` : '· all suites (scores only compare within a suite)';
  }

  renderTasks() {
    const el = this.root.querySelector('#tdTasks');
    if (!this.tasks.length) { el.innerHTML = ''; return; }
    el.innerHTML = this.tasks.map(t => {
      const ready = t.status === 'ready';
      const best = t.best_run ? `best <code>${esc(t.best_run.slice(-6))}</code>` : 'no run yet';
      return `<div class="td-task ${ready ? '' : 'td-task-blocked'}" title="${esc(t.goal)}\n${esc(t.status)}" data-suite="${esc(t.suite)}">
        <div class="td-task-name">${esc(t.task)} <span class="td-muted">→ ${esc(t.suite)}@${esc(t.suite_version || '?')}</span></div>
        <div class="td-muted">${ready ? '✅ ready' : '⛔ ' + esc(t.status.replace('blocked: ', ''))} · ${best}</div>
      </div>`;
    }).join('');
    el.querySelectorAll('.td-task').forEach(d => d.onclick = () => { this.suite = d.dataset.suite; this.refresh(); });
  }

  renderRuns() {
    const t = this.root.querySelector('#tdRuns');
    if (!this.runs.length) { t.innerHTML = '<tr><td class="td-muted">no runs yet</td></tr>'; return; }
    const head = `<tr><th></th><th>run</th><th>task / suite</th><th>status</th><th>gates</th><th>score</th><th>dist p50</th><th>falls</th><th>parent</th></tr>`;
    t.innerHTML = head + this.runs.map(r => {
      const busy = ['queued', 'training', 'benchmarking'].includes(r.status);
      const best = this.bestBySuite[r.suite]?.run_id === r.run_id ? ' 🥇' : '';
      const gates = r.gates_total ? `${r.gates_passed}/${r.gates_total}` : '—';
      const gcls = r.gates_total ? (r.gates_passed === r.gates_total ? 'td-ok' : 'td-warn') : '';
      const playing = r.run_id === this.playingRun ? ' <span class="td-playing" title="replaying in the viewer">▶ in viewer</span>' : '';
      return `<tr data-run="${esc(r.run_id)}" class="${r.run_id === this.selected ? 'td-row-sel' : ''}">
        <td><input type="checkbox" data-cmp="${esc(r.run_id)}" ${this.compare.has(r.run_id) ? 'checked' : ''}></td>
        <td><div class="td-runname">${esc(r.name)}${best}${playing}</div><code class="td-muted">${esc(r.run_id)}</code></td>
        <td>${esc(r.task || 'flat_walk')}<br><span class="td-muted">${esc(r.suite || 'flat_v1')}${r.suite_version ? '@' + esc(r.suite_version) : ''}</span></td>
        <td class="${busy ? 'td-busy' : (r.status === 'failed' ? 'td-bad' : '')}">${esc(r.status)}${r.error ? `<div class="td-bad td-tiny">${esc(String(r.error).slice(0, 80))}</div>` : ''}</td>
        <td class="${gcls}">${gates}</td><td>${fmt(r.score, 2)}</td><td>${fmt(r.dist_p50, 2)} m</td><td>${fmt(r.fall_rate, 2)}</td>
        <td class="td-muted"><code>${esc((r.parent || '—').slice(-6))}</code></td></tr>`;
    }).join('');
    // Clicking a run selects it AND replays its rollout (when it has a benchmark), so the viewer follows the selection.
    t.querySelectorAll('tr[data-run]').forEach(tr => tr.onclick = (e) => {
      if (e.target.type === 'checkbox') return;
      const id = tr.dataset.run;
      this.loadDetail(id);
      const r = this.runs.find(x => x.run_id === id);
      if (r && r.gates_total) this.replay(id, r.suite);
    });
    t.querySelectorAll('input[data-cmp]').forEach(cb => cb.onchange = () => {
      cb.checked ? this.compare.add(cb.dataset.cmp) : this.compare.delete(cb.dataset.cmp);
      this.root.querySelector('#tdCompareCount').textContent = `${this.compare.size} selected`;
      this.renderCompare();
    });
    const b = this.baselines || {};
    this.root.querySelector('#tdBaselines').innerHTML = Object.keys(b).length
      ? `firmware baselines on ${esc(this.baselinesSuite || 'flat_v1')}: ` + Object.entries(b).map(([k, v]) => `<code>${esc(k)}</code> ${fmt(v.distance_p50, 2)} m, falls ${fmt(v.fall_rate, 2)}`).join(' · ')
      : '';
  }

  // ── detail ─────────────────────────────────────────────────────────
  async loadDetail(runId, quiet = false) {
    this.selected = runId;
    if (!quiet) this.renderRuns();
    let d;
    try { d = await getJSON(`/api/training/run/${encodeURIComponent(runId)}`); } catch (e) {
      this.root.querySelector('#tdDetail').innerHTML = `<div class="td-bad">${esc(e.message)}</div>`; return;
    }
    this.lastDetail = d;
    const el = this.root.querySelector('#tdDetail');
    this.root.querySelector('#tdDetailTitle').textContent = `${d.name || runId} · ${d.task || 'flat_walk'} / ${d.suite || 'flat_v1'}`;
    const prog = d.progress || {};
    const pct = prog.total ? Math.round(100 * (prog.step || 0) / prog.total) : null;
    const bench = d.benchmark;
    const warm = d.warm_start || {};
    const m = d.metrics || {};
    const gates = (bench?.gates || []).filter(g => g.pass !== null);
    const diff = (d.config_diff || []);
    el.innerHTML = `
      <div class="td-kv">
        <span><b>status</b> ${esc(d.status)}${pct !== null && ['training', 'queued'].includes(d.status) ? ` · ${pct}%` : ''}</span>
        <span><b>steps</b> ${fmt((d.config?.ppo?.num_timesteps || 0) / 1e6, 0)}M · ${d.config?.ppo?.num_envs || '?'} envs</span>
        <span><b>start</b> ${warm.used ? 'warm from ' + esc((warm.parent || '').slice(-6)) : (warm.reason ? 'cold (' + esc(warm.reason) + ')' : 'from scratch')}</span>
        <span><b>train</b> ${m.elapsed_s ? fmt(m.elapsed_s / 60, 1) + ' min · ' + fmt(m.steps_per_s_mean, 0) + ' steps/s' : '—'}</span>
        <span><b>eval reward</b> ${fmt(m.reward_first, 0)} → ${fmt(m.reward_final, 0)}</span>
      </div>
      ${pct !== null && ['training'].includes(d.status) ? `<div class="td-progress"><div style="width:${pct}%"></div></div>` : ''}
      <div class="td-muted td-notes">${esc(d.notes || '')}</div>
      <canvas id="tdCurve" class="td-canvas"></canvas>
      ${bench ? `<div class="td-kv"><span><b>${esc(bench.suite)}@${esc(bench.suite_version)}</b></span><span class="${bench.passed ? 'td-ok' : 'td-warn'}"><b>${bench.gates_passed}/${bench.gates_total} gates</b> · score ${fmt(bench.score, 2)}</span>
          <span>${fmt(bench.metrics?.forward_distance_p50, 2)} m · falls ${fmt(bench.metrics?.fall_rate, 2)} · ${fmt(bench.metrics?.energy_proxy_w, 2)} W · peak joint ${fmt(bench.metrics?.peak_joint_speed_rad_s, 2)} rad/s</span></div>
        <canvas id="tdGateBars" class="td-canvas td-canvas-tall"></canvas>
        <details><summary>gate table (${gates.length})</summary><table class="td-table td-tiny">${gates.map(g => `<tr class="${g.pass ? 'td-ok' : 'td-bad'}"><td>${g.pass ? '✓' : '✗'}</td><td>${esc(g.gate)}</td><td>${fmt(g.value, 3)}</td><td>${esc(g.op)} ${Array.isArray(g.threshold) ? esc(JSON.stringify(g.threshold)) : esc(g.threshold)}</td><td class="td-muted">${esc(g.term || '')}</td></tr>`).join('')}</table></details>
        <div class="td-reflection">${esc(bench.reflection || '')}</div>` : '<div class="td-muted">no benchmark yet</div>'}
      <details ${diff.length ? '' : 'open'}><summary>config diff vs ${esc(d.parent ? 'parent ' + d.parent.slice(-6) : 'defaults')} (${diff.length})</summary>
        <table class="td-table td-tiny">${diff.map(x => `<tr><td><code>${esc(x.path)}</code></td><td class="td-muted">${esc(JSON.stringify(x.from))}</td><td>→ ${esc(JSON.stringify(x.to))}</td></tr>`).join('') || '<tr><td class="td-muted">no diff recorded</td></tr>'}</table></details>
      <div class="td-actions">
        <button class="vbtn td-small" id="tdReplay" ${bench ? '' : 'disabled'}>▶ replay rollout in viewer</button>
        <button class="vbtn td-small" id="tdCopyId">copy run id</button>
      </div>`;
    el.querySelector('#tdReplay').onclick = () => this.replay(runId, d.suite);
    el.querySelector('#tdCopyId').onclick = () => navigator.clipboard?.writeText(runId);
    this.drawDetailCharts(d);
  }

  drawDetailCharts(d) {
    const c = this.root.querySelector('#tdCurve');
    if (c) {
      const pts = d.curve || [];
      lineChart(c, [
        { label: 'eval reward', color: PALETTE[0], points: pts.map(p => [p.step / 1e6, p.reward]) },
      ], { title: 'training curve (eval episode reward vs env steps, M)', xfmt: v => fmt(v, 1) + 'M' });
    }
    const gb = this.root.querySelector('#tdGateBars');
    if (gb && d.benchmark) {
      const gates = (d.benchmark.gates || []).filter(g => g.pass !== null);
      const ratios = {}; gates.forEach(g => { ratios[g.gate] = gateRatio(g); });
      gateBars(gb, gates.map(g => g.gate), [{ label: d.name || d.run_id, color: PALETTE[0], ratios }], { title: 'gates: value / threshold (below the dashed line = pass)' });
    }
  }

  async replay(runId, suite) {
    if (!this.viewer || typeof this.viewer.playRollout !== 'function') { alert('3D viewer not available'); return; }
    const r = this.runs.find(x => x.run_id === runId);
    const label = `${r?.name || runId} · ${suite || r?.suite || ''}`.replace(/ · $/, '');
    const btn = this.root.querySelector('#tdReplay');
    const seqName = document.getElementById('timelineSeqName');
    if (btn) { btn.disabled = true; btn.textContent = '⏳ loading rollout…'; }
    if (seqName) seqName.textContent = `loading ${label}…`;
    const token = (this._replayToken = (this._replayToken || 0) + 1);
    try {
      const ro = await getJSON(`/api/training/rollout/${encodeURIComponent(runId)}?seed=0${suite ? '&suite=' + encodeURIComponent(suite) : ''}`);
      if (token !== this._replayToken) return; // a newer replay request superseded this one
      const ok = this.viewer.playRollout(ro, { loop: false, name: label });
      if (ok) { this.playingRun = runId; this.renderRuns(); }
    } catch (e) {
      if (seqName) seqName.textContent = 'rollout failed: ' + e.message;
      alert('rollout: ' + e.message);
    } finally {
      const b = this.root.querySelector('#tdReplay');
      if (b) { b.disabled = false; b.textContent = '▶ replay rollout in viewer'; }
    }
  }

  // ── compare ────────────────────────────────────────────────────────
  async renderCompare() {
    const el = this.root.querySelector('#tdCompare');
    const ids = [...this.compare];
    if (ids.length < 2) { el.innerHTML = '<span class="td-muted">Tick two or more runs in the leaderboard.</span>'; return; }
    let cmp, details;
    try {
      [cmp, details] = await Promise.all([
        getJSON('/api/training/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ run_ids: ids }) }),
        Promise.all(ids.map(id => getJSON(`/api/training/run/${encodeURIComponent(id)}`).catch(() => null))),
      ]);
    } catch (e) { el.innerHTML = `<div class="td-bad">${esc(e.message)}</div>`; return; }
    const suites = cmp.suites || {};
    const cross = cmp.cross_suite;
    const gatesSet = new Set(); details.forEach(d => (d?.benchmark?.gates || []).forEach(g => { if (g.pass !== null) gatesSet.add(g.gate); }));
    const gates = [...gatesSet];
    const groups = details.map((d, i) => {
      const ratios = {}; (d?.benchmark?.gates || []).forEach(g => { if (g.pass !== null) ratios[g.gate] = gateRatio(g); });
      return { label: d?.name || ids[i], color: PALETTE[i % PALETTE.length], ratios };
    });
    const rows = cmp.runs || {};
    el.innerHTML = `
      ${cross ? '<div class="td-warn">⚠ these runs are judged on different suites — gate values are not comparable across suites</div>' : ''}
      <table class="td-table td-tiny"><tr><th>run</th><th>suite</th><th>gates</th><th>score</th><th>dist</th><th>falls</th></tr>
        ${ids.map((id, i) => { const r = rows[id] || {}; return `<tr><td><span class="td-swatch" style="background:${PALETTE[i % PALETTE.length]}"></span>${esc(r.name || id)}</td><td>${esc(suites[id] || '')}</td><td>${r.gates_total ? `${r.gates_passed}/${r.gates_total}` : '—'}</td><td>${fmt(r.score, 2)}</td><td>${fmt(r.dist_p50, 2)}</td><td>${fmt(r.fall_rate, 2)}</td></tr>`; }).join('')}</table>
      <canvas id="tdCmpCurves" class="td-canvas"></canvas>
      <canvas id="tdCmpGates" class="td-canvas td-canvas-tall"></canvas>
      <details open><summary>gate values</summary><div class="td-tablewrap"><table class="td-table td-tiny"><tr><th>gate</th>${ids.map((id, i) => `<th style="color:${PALETTE[i % PALETTE.length]}">${esc((rows[id]?.name || id).slice(0, 18))}</th>`).join('')}</tr>
        ${Object.entries(cmp.gates_table || {}).map(([g, vals]) => `<tr><td>${esc(g)}</td>${ids.map(id => `<td>${fmt(vals[id], 3)}</td>`).join('')}</tr>`).join('')}</table></div></details>
      <details><summary>config diffs vs the first run</summary>${Object.entries(cmp.config_diffs_vs_first || {}).map(([id, diff]) => `<div><b>${esc(rows[id]?.name || id)}</b><table class="td-table td-tiny">${(diff || []).map(x => `<tr><td><code>${esc(x.path)}</code></td><td class="td-muted">${esc(JSON.stringify(x.from))}</td><td>→ ${esc(JSON.stringify(x.to))}</td></tr>`).join('') || '<tr><td class="td-muted">identical</td></tr>'}</table></div>`).join('')}</details>
      <details><summary>reflections</summary>${ids.map(id => `<div class="td-reflection"><b>${esc(rows[id]?.name || id)}:</b> ${esc(cmp.reflections?.[id] || '—')}</div>`).join('')}</details>`;
    lineChart(el.querySelector('#tdCmpCurves'), details.map((d, i) => ({ label: d?.name || ids[i], color: PALETTE[i % PALETTE.length], points: (d?.curve || []).map(p => [p.step / 1e6, p.reward]) })), { title: 'eval reward vs env steps (M) — each run scored by its OWN weights', xfmt: v => fmt(v, 1) + 'M' });
    gateBars(el.querySelector('#tdCmpGates'), gates, groups, { title: 'gates: value / threshold per run' });
  }

  redrawCharts() {
    if (this.lastDetail) this.drawDetailCharts(this.lastDetail);
    if (this.compare.size >= 2) this.renderCompare();
  }

  // ── GLM sessions ───────────────────────────────────────────────────
  async loadSessions() {
    try {
      const res = await getJSON('/api/training/sessions');
      this.sessions = res.sessions || [];
      this.liveSessionId = res.live;
      const sel = this.root.querySelector('#tdSessionPick');
      sel.innerHTML = this.sessions.length ? this.sessions.map(s => `<option value="${esc(s.session_id)}">${s.live ? '● LIVE · ' : ''}${esc(s.session_id)} · ${esc((s.prompt || '').slice(0, 50))}</option>`).join('') : '<option value="">no training sessions recorded yet</option>';
      const pick = res.live || (this.sessions[0] && this.sessions[0].session_id);
      if (pick && (!this.session || this.session.session_id !== pick)) await this.loadSession(pick);
      else if (!pick) this.root.querySelector('#tdSession').innerHTML = '<div class="td-muted">Start a training-mode chat in the GLM Agent tab (🧠 Training mode). Every cycle GLM runs shows up here, for every viewer, live.</div>';
    } catch (e) { /* trainer may be down; the panel stays */ }
  }

  async loadSession(id) {
    if (!id) return;
    try { this.session = await getJSON(`/api/training/sessions/${encodeURIComponent(id)}`); } catch (e) { return; }
    this.renderSession();
  }

  connectLive() {
    if (!window.EventSource) return;
    try { this.es = new EventSource('/api/training/sessions/live'); } catch (e) { return; }
    this.es.onmessage = (m) => {
      let msg; try { msg = JSON.parse(m.data); } catch (e) { return; }
      const live = this.root.querySelector('#tdLive');
      if (msg.type === 'hello' || msg.type === 'keepalive') {
        this.liveSessionId = msg.live;
        live.textContent = msg.live ? `● GLM session live: ${msg.live}` : 'no live GLM session';
        live.className = 'td-chip ' + (msg.live ? 'td-live' : '');
        if (msg.type === 'hello') this.loadSessions();
        return;
      }
      if (msg.type === 'session_start') { this.liveSessionId = msg.session.session_id; this.loadSessions(); live.textContent = `● GLM session live: ${msg.session.session_id}`; live.className = 'td-chip td-live'; return; }
      if (msg.type === 'session_end') { this.liveSessionId = null; live.textContent = 'no live GLM session'; live.className = 'td-chip'; this.loadSessions(); this.refresh(); return; }
      if (msg.type === 'event' && this.session && msg.session_id === this.session.session_id) {
        const ev = msg.event;
        const evs = this.session.events;
        if (['thought', 'text'].includes(ev.type) && evs.length && evs[evs.length - 1].type === ev.type) evs[evs.length - 1].content = (evs[evs.length - 1].content || '') + (ev.content || '');
        else evs.push(ev);
        if (msg.cycles) this.session.cycles = msg.cycles;
        this.session.live = true;
        this.renderSession();
        if (ev.type === 'tool_result') this.refresh();
      }
    };
  }

  renderSession() {
    const s = this.session; const el = this.root.querySelector('#tdSession');
    if (!s) return;
    const cycles = s.cycles || [];
    const cards = cycles.map((c, i) => {
      const a = c.args || {}; const r = c.result || {}; const p = (c.progress || {}).progress || {};
      const pct = p.total ? Math.round(100 * (p.step || 0) / p.total) : null;
      const patch = Object.keys(a.config_patch || {}).length ? JSON.stringify(a.config_patch) : '(no patch)';
      const status = c.status === 'running' ? `⏳ ${esc((c.progress || {}).status || 'running')}${pct !== null ? ' ' + pct + '%' : ''}${p.reward !== undefined && p.reward !== null ? ' · reward ' + fmt(p.reward, 0) : ''}` : (c.status === 'ok' ? '✓ done' : '✗ ' + esc(r.error || 'failed'));
      const res = r.run_id ? `<div class="td-cycle-res ${r.passed ? 'td-ok' : (r.gates_total ? 'td-warn' : '')}"><b>${esc(r.suite || '')}</b> ${r.gates_total ? `${r.gates_passed}/${r.gates_total} gates · score ${fmt(r.score, 2)}` : ''} · <code data-run="${esc(r.run_id)}" class="td-link">${esc(r.run_id.slice(-6))}</code></div>` : '';
      return `<div class="td-cycle ${c.status === 'running' ? 'td-cycle-live' : ''}">
        <div class="td-cycle-head">cycle ${i + 1} · <b>${esc(a.task || (c.result && c.result.task) || 'flat_walk')}</b> <span class="td-muted">${esc(c.tool)}</span></div>
        <div class="td-cycle-hyp">${esc(a.notes || a.name || '')}</div>
        <code class="td-cycle-patch">${esc(patch.slice(0, 220))}</code>
        ${a.base_run_id ? `<div class="td-muted td-tiny">warm start from <code>${esc(a.base_run_id.slice(-6))}</code></div>` : ''}
        ${pct !== null && c.status === 'running' ? `<div class="td-progress"><div style="width:${pct}%"></div></div>` : ''}
        <div class="td-cycle-status">${status}${c.t_end ? ` · ${fmt((c.t_end - c.t_start) / 60, 1)} min` : ''}</div>
        ${res}
        ${r.reflection ? `<details><summary>reflection</summary><div class="td-reflection">${esc(r.reflection)}</div></details>` : ''}
      </div>`;
    }).join('');
    const feed = (s.events || []).slice(-40).map(ev => {
      if (ev.type === 'thought') return `<details class="td-ev"><summary>🧠 thinking (${(ev.content || '').length} chars)</summary><div class="td-muted td-pre">${esc(ev.content)}</div></details>`;
      if (ev.type === 'text') return `<div class="td-ev">💬 ${esc(ev.content)}</div>`;
      if (ev.type === 'tool_call') return `<div class="td-ev td-muted">⚙️ ${esc(ev.name)} <code>${esc(JSON.stringify(ev.args || {}).slice(0, 140))}</code></div>`;
      if (ev.type === 'tool_result') { const r = ev.result || {}; return `<div class="td-ev ${r.ok ? '' : 'td-bad'}">${r.ok ? '✓' : '✗'} ${esc(ev.name || '')}: ${esc((r.reflection || r.error || r.advice || (r.tasks ? r.tasks.length + ' tasks' : '') || 'ok').toString().slice(0, 200))}</div>`; }
      if (ev.type === 'done') return `<div class="td-ev td-ok">■ session finished${ev.final_message ? ': ' + esc(ev.final_message.slice(0, 300)) : ''}</div>`;
      if (ev.type === 'error') return `<div class="td-ev td-bad">✗ ${esc(ev.content || ev.detail || ev.error || 'error')}</div>`;
      return '';
    }).join('');
    const started = s.started ? new Date(s.started * 1000).toLocaleString() : '';
    el.innerHTML = `
      <div class="td-kv"><span>${s.live ? '<span class="td-live">● LIVE</span>' : 'finished'}</span><span><b>prompt</b> ${esc(s.prompt || '')}</span><span class="td-muted">${started} · ${s.n_events || (s.events || []).length} events · ${cycles.length} training cycles</span></div>
      <div class="td-cycles">${cards || '<div class="td-muted">no training cycle yet in this session</div>'}</div>
      <details ${s.live ? 'open' : ''}><summary>event feed (last ${Math.min(40, (s.events || []).length)})</summary><div class="td-feed">${feed}</div></details>`;
    el.querySelectorAll('code[data-run]').forEach(c => c.onclick = () => this.loadDetail(c.dataset.run));
    if (s.live) { const f = el.querySelector('.td-feed'); if (f) f.scrollTop = f.scrollHeight; }
  }
}
