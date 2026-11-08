(() => {
  "use strict";
  // Exact, deterministic teaching reference. No learned FB or learner evidence.
  const N = 5, actions = [-1, 0, 1], names = ["left", "stay", "right"], arrows = ["←", "·", "→"];
  const dot = (a, b) => a.reduce((v, x, i) => v + x * b[i], 0);
  const next = (s, a) => Math.max(0, Math.min(N - 1, s + actions[a]));
  const reward = (goal) => Array.from({length: N}, (_, s) => +(s === goal));
  const fmt = (v) => (Math.abs(v) < 0.00005 ? 0 : v).toFixed(3);
  const vec = (v) => "[" + v.map(fmt).join(", ") + "]";
  const $ = (id) => document.getElementById(id);

  function inverse(matrix) {
    const rows = matrix.map((r, i) => [...r, ...Array.from({length: N}, (_, j) => +(i === j))]);
    for (let k = 0; k < N; k++) {
      let p = k;
      for (let i = k + 1; i < N; i++) if (Math.abs(rows[i][k]) > Math.abs(rows[p][k])) p = i;
      [rows[k], rows[p]] = [rows[p], rows[k]];
      const pivot = rows[k][k];
      if (Math.abs(pivot) < 1e-12) throw new Error("Singular reference matrix");
      rows[k] = rows[k].map((x) => x / pivot);
      for (let i = 0; i < N; i++) if (i !== k) {
        const v = rows[i][k];
        rows[i] = rows[i].map((x, j) => x - v * rows[k][j]);
      }
    }
    return rows.map((r) => r.slice(N));
  }

  function occupancy(policy, gamma) {
    return inverse(Array.from({length: N}, (_, s) => Array.from({length: N}, (_, j) =>
      +(s === j) - gamma * +(next(s, policy[s]) === j))));
  }
  const values = (policy, r, gamma) => occupancy(policy, gamma).map((row) => dot(row, r));
  const qCurrent = (s, a, v, r, gamma) => r[s] + gamma * v[next(s, a)];
  function greedy(q) {
    // Prefer stay on exact ties so a flat reward does not imply a goal preference.
    let best = 1;
    for (const a of [0, 2]) if (q[a] > q[best] + 1e-9) best = a;
    return best;
  }
  function optimal(r, gamma) {
    let v = Array(N).fill(0);
    for (let i = 0; i < 1000; i++) {
      const vnew = v.map((_, s) => Math.max(...actions.map((_, a) => qCurrent(s, a, v, r, gamma))));
      const delta = Math.max(...v.map((x, s) => Math.abs(x - vnew[s])));
      v = vnew;
      if (delta < 1e-12) break;
    }
    return {v, policy: v.map((_, s) => greedy(actions.map((__, a) => qCurrent(s, a, v, r, gamma))))};
  }
  function gpi(goal, gamma = 0.5) {
    const r = reward(goal), vr = values(Array(N).fill(2), r, gamma), vl = values(Array(N).fill(0), r, gamma);
    const q = Array.from({length: N}, (_, s) => actions.map((_, a) =>
      Math.max(qCurrent(s, a, vr, r, gamma), qCurrent(s, a, vl, r, gamma))));
    const policy = q.map(greedy);
    return {r, vr, vl, q, policy, envelope: vr.map((v, s) => Math.max(v, vl[s])), actual: values(policy, r, gamma)};
  }
  function basis(d) {
    // Orthonormal eigenvectors of the unweighted five-node path graph Laplacian.
    return Array.from({length: N}, (_, s) => Array.from({length: d}, (_, k) =>
      k === 0 ? 1 : Math.sqrt(2) * Math.cos(Math.PI * k * (s + 0.5) / N)));
  }
  function gpiRollout(goal, start, continuation = "gpi") {
    const g = gpi(goal), rows = []; let s = start, partial = 0;
    for (let t = 0; t < 8; t++) {
      const a = t === 0 || continuation === "gpi" ? g.policy[s] : continuation === "right" ? 2 : 0;
      const contribution = 0.5 ** t * g.r[s]; partial += contribution;
      rows.push({t, s, a, reward: g.r[s], discount: 0.5 ** t, contribution, partial});
      s = next(s, a);
    }
    const total = continuation === "gpi" ? g.actual[start] :
      qCurrent(start, g.policy[start], continuation === "right" ? g.vr : g.vl, g.r, 0.5);
    return {rows, total, tail: total - partial};
  }
  function project(goal, d) {
    const b = basis(d), r = reward(goal);
    const z = Array.from({length: d}, (_, k) => dot(b.map((row) => row[k]), r) / N);
    const rhat = b.map((row) => dot(row, z)), solved = optimal(rhat, 0.5);
    return {b, r, z, rhat, policy: solved.policy, m: occupancy(solved.policy, 0.5)};
  }

  // Exposed pure reference functions also allow independent numeric verification.
  window.ZeroShotRLReference = {next, occupancy, values, optimal, gpi, gpiRollout, basis, project};

  function table(headers, rows) {
    return "<table><thead><tr>" + headers.map((x) => "<th scope=\"col\">" + x + "</th>").join("") +
      "</tr></thead><tbody>" + rows.map((r) => "<tr>" + r.map((x, i) => i === 0 ?
        "<th scope=\"row\">" + x + "</th>" : "<td>" + x + "</td>").join("") + "</tr>").join("") + "</tbody></table>";
  }
  function corridor(id, policy, goal) {
    $(id).innerHTML = Array.from({length: N}, (_, s) => {
      const x = 25 + 125 * s, isGoal = s === goal;
      return `<rect x="${x}" y="15" width="110" height="80" rx="9" fill="${isGoal ? "#daf0df" : "#eaf2f6"}" stroke="${isGoal ? "#35754c" : "#b5ccd5"}"/>` +
        `<text x="${x + 55}" y="38" text-anchor="middle">格 ${s}${isGoal ? " · reward" : ""}</text>` +
        `<text x="${x + 55}" y="76" text-anchor="middle" style="font-size:30px">${arrows[policy[s]]}</text>`;
    }).join("");
  }
  function renderSR() {
    const gamma = +$("sr-gamma").value, goal = +$("sr-goal").value, p = $("sr-policy").value;
    const policy = Array.from({length: N}, (_, s) => p === "right" ? 2 : p === "left" ? 0 : s < 2 ? 2 : s > 2 ? 0 : 1);
    const m = occupancy(policy, gamma), r = reward(goal), v = m.map((row) => dot(row, r)), mass = 1 / (1 - gamma);
    $("sr-gamma-label").textContent = gamma.toFixed(2);
    corridor("sr-corridor", policy, goal);
    $("sr-matrix").innerHTML = table(["起点 / 终点", "0", "1", "2", "3", "4", "行和", "V = Mr"], m.map((row, s) => [s,
      ...row.map((x) => `<span style="display:block;border-radius:4px;background:rgba(18,101,120,${(0.05 + 0.55 * x / mass).toFixed(3)})">${fmt(x)}</span>`),
      fmt(row.reduce((a, b) => a + b, 0)), `<span class="winner">${fmt(v[s])}</span>`]));
    $("sr-stats").innerHTML = `<div>当前策略价值 V<strong>${vec(v)}</strong></div><div>每行总访问量 1 / (1 − γ)<strong>${fmt(mass)}</strong></div>`;
  }
  function renderGPI() {
    const goal = +$("gpi-goal").value, start = +$("gpi-start").value, g = gpi(goal);
    corridor("gpi-corridor", g.policy, goal);
    $("gpi-q").innerHTML = table(["起点 " + start, "left", "stay", "right"], [
      ["QπR", ...actions.map((_, a) => fmt(qCurrent(start, a, g.vr, g.r, 0.5)))],
      ["QπL", ...actions.map((_, a) => fmt(qCurrent(start, a, g.vl, g.r, 0.5)))],
      ["maxᵢ Qπi", ...g.q[start].map((q, a) => `<span class="${a === g.policy[start] ? "winner" : ""}">${fmt(q)}${a === g.policy[start] ? " ✓" : ""}</span>`)]
    ]);
    $("gpi-values").innerHTML = table(["起点", "VR", "VL", "max(VR,VL)", "执行 GPI 的 V"], g.vr.map((v, s) =>
      [s, fmt(v), fmt(g.vl[s]), fmt(g.envelope[s]), `<span class="winner">${fmt(g.actual[s])}</span>`]));
    const mode = $("gpi-continuation").value, trace = gpiRollout(goal, start, mode);
    $("gpi-trace").textContent = `状态时间线：${trace.rows.map((row) => row.s).join(" → ")} → …；${mode === "gpi" ? "每步重新选择 GPI 动作" : "只做第一步后，一直跟随 " + (mode === "right" ? "πR" : "πL")}。`;
    $("gpi-timeline").innerHTML = table(["时刻", "状态", "当前 reward", "之后的动作", "折扣", "本次贡献"],
      trace.rows.map((row) => ["t=" + row.t, row.s, row.reward, names[row.a], fmt(row.discount), fmt(row.contribution)]));
    $("gpi-total").textContent = `表中前八项之和 = ${fmt(trace.rows[7].partial)}；剩余无限尾项 = ${fmt(trace.tail)}；完整 return = ${fmt(trace.total)}。这属于${mode === "gpi" ? "新 GPI 策略的实际价值" : "首动作后跟随旧策略的 Q 估值"}。`;
  }
  function renderProjection() {
    const goal = +$("proj-goal").value, d = +$("proj-rank").value, s = +$("proj-start").value, a = +$("proj-action").value;
    const p = project(goal, d), occ = p.m[next(s, a)], f = Array.from({length: d}, (_, k) => dot(occ, p.b.map((row) => row[k])));
    const occHat = p.b.map((row) => dot(f, row) / N), qhat = dot(f, p.z), qtrue = dot(occ, p.r);
    const baseY = 170, scale = 120;
    let bars = `<line x1="20" y1="${baseY}" x2="640" y2="${baseY}" stroke="#90aab5"/><text x="22" y="22">蓝：原 reward r　橙：投影 reward r̂</text>`;
    for (let i = 0; i < N; i++) {
      const x = 48 + 123 * i;
      [p.r[i], p.rhat[i]].forEach((v, k) => {
        const height = Math.abs(v) * scale, y = v >= 0 ? baseY - height : baseY;
        bars += `<rect x="${x + 29 * k}" y="${y}" width="24" height="${height}" fill="${k === 0 ? "#267b9a" : "#b76b25"}"/><text x="${x + 12 + 29 * k}" y="${v >= 0 ? y - 7 : y + height + 17}" text-anchor="middle" style="font-size:12px">${fmt(v)}</text>`;
      });
      bars += `<text x="${x + 25}" y="228" text-anchor="middle">格 ${i}</text>`;
    }
    $("proj-bars").innerHTML = bars;
    corridor("proj-corridor", p.policy, goal);
    $("proj-vectors").textContent = `z = ${vec(p.z)}；F(s=${s},a=${names[a]},z) = ${vec(f)}；Fᵀz = ${fmt(qhat)}；该策略原 reward 下的 Q = ${fmt(qtrue)}。`;
    $("proj-values").innerHTML = table(["状态 j", "原 reward", "重构 reward", "真实 Mπ(s,a,j)", "FᵀB(j)ρ(j)"],
      p.r.map((r, j) => [j, fmt(r), fmt(p.rhat[j]), fmt(occ[j]), fmt(occHat[j])]));
  }
  ["sr-policy", "sr-goal", "sr-gamma"].forEach((id) => $(id).addEventListener("input", renderSR));
  ["gpi-goal", "gpi-start", "gpi-continuation"].forEach((id) => $(id).addEventListener("input", renderGPI));
  ["proj-goal", "proj-rank", "proj-start", "proj-action"].forEach((id) => $(id).addEventListener("input", renderProjection));
  renderSR(); renderGPI(); renderProjection();
})();
