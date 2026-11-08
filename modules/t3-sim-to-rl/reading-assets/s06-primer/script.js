(() => {
  const TOL = 0.05, ENV = 0.02, GOOD = 0.0, BAD = 0.30, SD = 0.08, XMIN = -0.6, XMAX = 0.75;
  const $ = (id) => document.getElementById(id);
  const PAL = {"--truth": "#38724b", "--critic": "#a34f08", "--base": "#1d6fb8", "--edit": "#8b375d", "--pick": "#111827", "--band": "rgba(56,114,75,.12)", "--line": "#d9e4e9", "--muted": "#536777", "--text": "#213341", "--surface-2": "#eef3f5"};
  const css = (name) => PAL[name];

  let seed = 7;
  const rand = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
  const randn = () => { let u = 0, v = 0; while (u === 0) u = rand(); v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  const truth = (e) => Phi((TOL - e) / ENV) - Phi((-TOL - e) / ENV);
  const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

  const P = () => ({
    pgood: +$("s6-pgood").value, n: 2 ** +$("s6-nexp").value, beta: +$("s6-beta").value,
    err: +$("s6-err").value, reducer: $("s6-reducer").value,
  });
  const density = (e, p) => p.pgood * npdf(e, GOOD, SD) + (1 - p.pgood) * npdf(e, BAD, SD);
  const sampleBase = (p) => (rand() < p.pgood ? GOOD : BAD) + SD * randn();

  // a critic = 10 smooth random error fields; amplitude grows where VLA density is low
  const newField = () => Array.from({ length: 4 }, () => ({ a: randn(), f: 4 + 10 * rand(), ph: 6.28 * rand() }));
  const newCritic = () => Array.from({ length: 10 }, newField);
  const fieldAt = (fd, e) => fd.reduce((s, c) => s + c.a * Math.sin(c.f * e + c.ph), 0) / 2;
  let critic = newCritic();
  let pickIdx = [0, 1];
  const qhat = (e, p, cr = critic, idx = pickIdx) => {
    const dmax = density(GOOD, p) || 1;
    const support = Math.min(1, density(e, p) / dmax);
    const amp = p.err * (0.35 + 1.65 * (1 - support));
    const vals = (p.reducer === "single" ? [idx[0]] : idx).map((k) => truth(e) + amp * fieldAt(cr[k], e));
    return Math.min(...vals);
  };
  const editOf = (e, p, cr, idx) => {
    if (p.beta <= 0) return e;
    let best = e, bv = qhat(e, p, cr, idx);
    for (let i = -20; i <= 20; i++) { const c = e + (p.beta * i) / 20; const v = qhat(c, p, cr, idx); if (v > bv) { bv = v; best = c; } }
    return best;
  };
  const choose = (p, cr = critic, idx = pickIdx) => {
    const base = Array.from({ length: p.n }, () => sampleBase(p));
    const edits = base.map((e) => editOf(e, p, cr, idx));
    const cands = [...base.map((e) => ({ e, edit: false })), ...(p.beta > 0 ? edits.map((e, i) => ({ e, edit: true, from: base[i] })) : [])];
    cands.forEach((c) => { c.q = qhat(c.e, p, cr, idx); c.t = truth(c.e); });
    const pick = cands.reduce((a, b) => (b.q > a.q ? b : a));
    return { cands, pick };
  };

  let current = null;
  const fmt = (x, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : "–");
  const stat = (k, v) => `<div>${k}<strong>${v}</strong></div>`;

  function drawA() {
    const p = P();
    $("s6-v-pgood").textContent = p.pgood.toFixed(2); $("s6-v-n").textContent = p.n;
    $("s6-v-beta").textContent = p.beta.toFixed(2); $("s6-v-err").textContent = p.err.toFixed(2);
    const W = 720, H = 360, L = 48, R = 16, T = 16, B = 40;
    const sx = (e) => L + ((e - XMIN) / (XMAX - XMIN)) * (W - L - R);
    const sy = (v) => T + (1 - (v + 0.25) / 1.5) * (H - T - B);
    const grid = []; for (let i = 0; i <= 270; i++) grid.push(XMIN + ((XMAX - XMIN) * i) / 270);
    const dmax = Math.max(...grid.map((e) => density(e, p)));
    const path = (f) => grid.map((e, i) => `${i ? "L" : "M"}${sx(e).toFixed(1)},${sy(f(e)).toFixed(1)}`).join("");
    const dens = `M${sx(XMIN)},${sy(-0.25)}` + grid.map((e) => `L${sx(e).toFixed(1)},${sy(-0.25 + 0.7 * density(e, p) / dmax).toFixed(1)}`).join("") + `L${sx(XMAX)},${sy(-0.25)}Z`;
    let s = `<rect x="${sx(-TOL)}" y="${T}" width="${sx(TOL) - sx(-TOL)}" height="${H - T - B}" fill="${css("--band")}"/>`;
    s += `<path d="${dens}" fill="${css("--base")}" opacity=".18"/>`;
    [0, 0.5, 1].forEach((v) => { s += `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}" stroke="${css("--line")}" stroke-dasharray="3 4"/><text x="${L - 8}" y="${sy(v) + 4}" text-anchor="end" font-size="11" fill="${css("--muted")}">${v}</text>`; });
    [-0.6, -0.3, 0, 0.3, 0.6].forEach((e) => { s += `<text x="${sx(e)}" y="${H - B + 18}" text-anchor="middle" font-size="11" fill="${css("--muted")}">${e}</text>`; });
    s += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle" font-size="12" fill="${css("--muted")}">落点误差 x'（成功带 ±0.05）</text>`;
    s += `<path d="${path(truth)}" fill="none" stroke="${css("--truth")}" stroke-width="2.4"/>`;
    s += `<path d="${path((e) => qhat(e, p))}" fill="none" stroke="${css("--critic")}" stroke-width="2"/>`;
    if (current) {
      current.cands.forEach((c) => {
        const x = sx(c.e), y = sy(c.q);
        if (c.edit) {
          s += `<line x1="${sx(c.from)}" x2="${x}" y1="${y}" y2="${y}" stroke="${css("--edit")}" stroke-width="1" opacity=".6"/>`;
          s += `<path d="M${x},${y - 5}L${x + 5},${y}L${x},${y + 5}L${x - 5},${y}Z" fill="${css("--edit")}" opacity=".85"/>`;
        } else {
          s += `<circle cx="${x}" cy="${y}" r="4.2" fill="${css("--base")}" opacity=".85"/>`;
        }
      });
      const pk = current.pick;
      s += `<circle cx="${sx(pk.e)}" cy="${sy(pk.q)}" r="9" fill="none" stroke="${css("--pick")}" stroke-width="2"/>`;
      s += `<line x1="${sx(pk.e)}" x2="${sx(pk.e)}" y1="${sy(pk.q)}" y2="${sy(pk.t)}" stroke="${css("--pick")}" stroke-dasharray="2 3"/>`;
      s += `<circle cx="${sx(pk.e)}" cy="${sy(pk.t)}" r="4" fill="${css("--truth")}"/>`;
    }
    $("s6-plot").innerHTML = s;
    if (current) {
      const pk = current.pick;
      $("s6-stats-one").innerHTML = stat("被选中候选的 critic 评分", fmt(pk.q)) + stat("它的真实成功概率", fmt(pk.t)) +
        stat("评分 − 真实（这一次）", fmt(pk.q - pk.t)) + stat("选中的是", pk.edit ? "edit 候选" : "base 候选");
    }
  }

  function runTrials() {
    const p = P(), M = 2000;
    let tSum = 0, gap = 0, ed = 0, bound = 0, oracle = 0, single = 0;
    const saved = seed;
    for (let i = 0; i < M; i++) {
      const cr = newCritic(); const idx = [Math.floor(rand() * 10), Math.floor(rand() * 10)];
      const { cands, pick } = choose(p, cr, idx);
      tSum += pick.t; gap += pick.q - pick.t; ed += pick.edit ? 1 : 0;
      if (pick.edit && Math.abs(pick.e - pick.from) >= p.beta * 0.95) bound += 1;
      oracle += Math.max(...cands.map((c) => c.t)); single += truth(sampleBase(p));
    }
    seed = saved;
    $("s6-stats-many").innerHTML = stat("单次 VLA 采样的成功率", fmt(single / M)) + stat("OTF 选中者的真实成功率", fmt(tSum / M)) +
      stat("候选中最好者（完美 critic）", fmt(oracle / M)) + stat("平均高估：评分 − 真实", fmt(gap / M)) +
      stat("选中 edit 的比例", fmt(ed / M, 2)) + stat("其中 edit 贴着 β 上限", fmt(ed ? bound / ed : NaN, 2));
  }

  function drawB() {
    const p = P(), M = 2000, saved = seed;
    let sSingle = 0, sOracle = 0, sOtf = 0;
    for (let i = 0; i < M; i++) {
      sSingle += truth(sampleBase(p));
      const cr = newCritic(); const idx = [Math.floor(rand() * 10), Math.floor(rand() * 10)];
      const { cands, pick } = choose(p, cr, idx);
      sOracle += Math.max(...cands.map((c) => c.t)); sOtf += pick.t;
    }
    seed = saved;
    const bars = [
      ["a' = 1 个 VLA 样本（SARSA 式）", sSingle / M, css("--base")],
      ["a' = 2N 候选中真实最好者", sOracle / M, css("--truth")],
      ["a' = OTF（learned critic 选）", sOtf / M, css("--critic")],
      ["a' = 全动作空间最优（V*）", truth(0), css("--muted")],
    ];
    const W = 720, L = 250, R = 60, rowH = 48;
    let s = "";
    bars.forEach(([name, v, c], i) => {
      const y = 14 + i * rowH, w = (W - L - R) * v;
      s += `<text x="${L - 10}" y="${y + 20}" text-anchor="end" font-size="13" fill="${css("--text")}">${name}</text>`;
      s += `<rect x="${L}" y="${y + 4}" width="${W - L - R}" height="24" rx="5" fill="${css("--surface-2")}"/>`;
      s += `<rect x="${L}" y="${y + 4}" width="${Math.max(0, w)}" height="24" rx="5" fill="${c}" opacity=".85"/>`;
      s += `<text x="${L + Math.max(0, w) + 8}" y="${y + 21}" font-size="13" fill="${css("--text")}">${v.toFixed(3)}</text>`;
    });
    $("s6-bars").innerHTML = s;
  }

  const refresh = (resample) => { if (resample || !current) current = choose(P()); drawA(); };
  ["s6-pgood", "s6-nexp", "s6-beta", "s6-err", "s6-reducer"].forEach((id) => $(id).addEventListener("input", () => { refresh(true); $("s6-stats-many").innerHTML = ""; drawB(); }));
  $("s6-resample").onclick = () => refresh(true);
  $("s6-newcritic").onclick = () => { critic = newCritic(); pickIdx = [Math.floor(rand() * 10), Math.floor(rand() * 10)]; refresh(true); };
  $("s6-trials").onclick = runTrials;
    refresh(true); drawB();
})();
