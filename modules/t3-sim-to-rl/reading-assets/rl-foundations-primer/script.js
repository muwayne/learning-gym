(() => {
  let seed = 11;
  const rand = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
  const randn = () => { let u = 0; while (u === 0) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()); };
  const $ = (id) => document.getElementById(id);
  const NS = [1, 2, 4, 8, 16, 32, 64];
  function sim(n, sigma, caseB, M = 4000) {
    const out = { single: [0, 0], double: [0, 0], min2: [0, 0] };
    for (let m = 0; m < M; m++) {
      const truth = Array.from({ length: n }, (_, i) => (caseB && i === 0 ? 0.6 : 0.5));
      const A = truth.map((t) => t + sigma * randn());
      const B = truth.map((t) => t + sigma * randn());
      const ens = truth.map((t) => Array.from({ length: 10 }, () => t + sigma * randn()));
      let ia = 0; A.forEach((v, i) => { if (v > A[ia]) ia = i; });
      out.single[0] += A[ia] - truth[ia]; out.single[1] += ia === 0 ? 1 : 0;
      out.double[0] += B[ia] - truth[ia]; out.double[1] += ia === 0 ? 1 : 0;
      const mean = ens.map((e) => e.reduce((a, b) => a + b, 0) / 10);
      let im = 0; mean.forEach((v, i) => { if (v > mean[im]) im = i; });
      const j = Math.floor(rand() * 10); let k = Math.floor(rand() * 9); if (k >= j) k += 1;
      out.min2[0] += Math.min(ens[im][j], ens[im][k]) - truth[im]; out.min2[1] += im === 0 ? 1 : 0;
    }
    for (const key in out) out[key] = out[key].map((v) => v / M);
    return out;
  }
  function draw() {
    const sigma = +$("mb-sigma").value, caseB = $("mb-case").value === "B", nIdx = +$("mb-n").value;
    $("mb-v-sigma").textContent = sigma.toFixed(2); $("mb-v-n").textContent = NS[nIdx];
    const saved = seed; seed = 11;
    const res = NS.map((n) => sim(n, sigma, caseB, 2000));
    seed = saved;
    const W = 720, H = 300, L = 56, R = 20, T = 18, B = 40;
    const vals = res.flatMap((r) => [r.single[0], r.double[0], r.min2[0]]);
    const lo = Math.min(-0.05, ...vals), hi = Math.max(0.05, ...vals);
    const sx = (i) => L + (i / (NS.length - 1)) * (W - L - R);
    const sy = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    let s = `<line x1="${L}" x2="${W - R}" y1="${sy(0)}" y2="${sy(0)}" stroke="#8094a1"/>`;
    const ticks = 4; for (let t = 0; t <= ticks; t++) { const v = lo + ((hi - lo) * t) / ticks; s += `<text x="${L - 8}" y="${sy(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`; }
    NS.forEach((n, i) => { s += `<text x="${sx(i)}" y="${H - B + 18}" text-anchor="middle">${n}</text>`; });
    s += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">候选数 N</text>`;
    s += `<text x="14" y="${T + 6}">高估量</text>`;
    [["single", "#a34f08"], ["double", "#1d6fb8"], ["min2", "#38724b"]].forEach(([k, c]) => {
      s += `<polyline points="${res.map((r, i) => `${sx(i)},${sy(r[k][0])}`).join(" ")}" fill="none" stroke="${c}" stroke-width="3"/>`;
      res.forEach((r, i) => { s += `<circle cx="${sx(i)}" cy="${sy(r[k][0])}" r="${i === nIdx ? 6 : 3.5}" fill="${c}"/>`; });
    });
    $("mb-plot").innerHTML = s;
    const r = res[nIdx];
    const cell = (k, v) => `<div>${k}<strong>${v}</strong></div>`;
    let h = cell("single 高估量", r.single[0].toFixed(3)) + cell("double 高估量", r.double[0].toFixed(3)) + cell("min of 2 高估量", r.min2[0].toFixed(3));
    if (caseB) h += cell("single / double 挑中真正最好者", r.single[1].toFixed(2)) + cell("min of 2（按均值挑）挑中最好者", r.min2[1].toFixed(2)) + cell("随机挑的基准", (1 / NS[nIdx]).toFixed(2));
    $("mb-stats").innerHTML = h;
  }
  ["mb-sigma", "mb-case", "mb-n"].forEach((id) => $(id).addEventListener("input", draw));
  draw();
})();
