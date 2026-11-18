"use strict";

const lab = document.getElementById("backward-lab");
const weightButtons = Array.from(lab.querySelectorAll("[data-weight]"));
const phaseButtons = Array.from(lab.querySelectorAll("[data-phase]"));
const summary = document.getElementById("lab-summary");
let weight = 0.25;
let phase = "forward";

function trace(w) {
  const scaled = 2 * w;
  const clipped = Math.min(1, Math.max(-1, scaled));
  const minimum = Math.min(w, 0.75);
  const residual1 = clipped - 0.8;
  const residual2 = minimum - 1;
  const error1 = residual1 ** 2;
  const error2 = residual2 ** 2;
  const clipSlope = Number(-1 < scaled && scaled < 1);
  const minSlope = w === 0.75 ? 0.5 : Number(w < 0.75);
  const gradScaled = residual1 * clipSlope;
  const gradInput = residual2 * minSlope;
  return {
    scaled, clipped, minimum, error1, error2, clipSlope, minSlope,
    residual1, residual2, gradScaled, gradInput,
    loss: (error1 + error2) / 2,
    gradWeight: 2 * gradScaled + gradInput,
  };
}

const format = value => Number(value.toFixed(8)).toString();
const gradient = value => "此值收到的梯度 " + format(value);

function render() {
  const t = trace(weight);
  const rows = {
    weight: [weight, "路径 1 " + format(2 * t.gradScaled) + " + 路径 2 (" +
      format(t.gradInput) + ") = " + format(t.gradWeight)],
    scale: [t.scaled, gradient(t.gradScaled)],
    input: [weight, gradient(t.gradInput)],
    clip: [t.clipped, gradient(t.residual1) + " · 局部斜率 " + t.clipSlope],
    min: [t.minimum, gradient(t.residual2) + " · 对 w 的局部斜率 " + t.minSlope],
    error1: [t.error1, gradient(0.5)],
    error2: [t.error2, gradient(0.5)],
    loss: [t.loss, "反传起点 ∂loss/∂loss = 1"],
  };
  for (const [name, [value, detail]] of Object.entries(rows)) {
    const node = lab.querySelector('[data-node="' + name + '"]');
    node.querySelector("[data-value]").textContent = format(value);
    const annotation = node.querySelector("[data-grad]");
    annotation.textContent = detail;
    annotation.hidden = phase === "forward";
  }
  const backward = phase !== "forward";
  lab.dataset.phase = phase;
  lab.querySelector('[data-node="clip"]').dataset.zeroSlope = String(backward && t.clipSlope === 0);
  lab.querySelector('[data-node="min"]').dataset.zeroSlope = String(backward && t.minSlope === 0);
  for (const button of weightButtons) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.weight) === weight));
  }
  for (const button of phaseButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.phase === phase));
  }
  if (phase === "forward") {
    summary.textContent = "从上往下算。两条平方误差汇入 mean，得到一个标量 loss。";
  } else if (phase === "backward") {
    summary.textContent = "从下往上看。mean 给两条平方误差各传 0.5。各路径沿局部导数反传，最后在 w 相加。";
  } else {
    const next = weight - 0.1 * t.gradWeight;
    summary.textContent = "学习率 0.1。用 w.grad = " + format(t.gradWeight) +
      " 更新 w，" + format(weight) + " → " + format(next) +
      "。重新 forward 后，loss " + format(t.loss) + " → " + format(trace(next).loss) + "。";
  }
}

for (const button of weightButtons) {
  button.addEventListener("click", () => {
    weight = Number(button.dataset.weight);
    render();
  });
}
for (const button of phaseButtons) {
  button.addEventListener("click", () => {
    phase = button.dataset.phase;
    render();
  });
}
render();
