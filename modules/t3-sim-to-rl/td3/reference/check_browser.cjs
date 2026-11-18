// Usage: node check_browser.cjs. No browser or dependencies required.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(require('node:path').join(__dirname, '../training-debugger.html'), 'utf8');
const code = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
const context = vm.createContext({structuredClone});
vm.runInContext(code + '\nglobalThis.frames = makeTrace();', context);
const frames = JSON.parse(JSON.stringify(context.frames));
const target = frames.find(f => f.phase === 'Update 1 · Target');
assert.ok(Math.abs(target.targetDetail.target.y[0] - 0.058829) < 5e-7);
for (const update of [1, 2]) {
  const forward = frames.find(f => f.phase === `Update ${update} · Critic forward`);
  const backward = frames.find(f => f.phase === `Update ${update} · Critic backward`);
  assert.deepEqual(forward.models, backward.models);
}
const forward = frames.find(f => f.phase.endsWith('Actor forward'));
const backward = frames.find(f => f.phase.endsWith('Actor backward'));
const step = frames.find(f => f.phase.endsWith('Actor step'));
assert.deepEqual(forward.models, backward.models);
assert.deepEqual(step.models.q1, backward.models.q1);
assert.notDeepEqual(step.models.actor, backward.models.actor);
const sync = frames.find(f => f.phase.endsWith('Target sync'));
assert.ok(Math.abs(sync.models.q2Target.b - 0.031888) < 5e-7);
console.log(JSON.stringify({frames: frames.length, target: target.targetDetail.target.y[0],
  q2TargetBias: sync.models.q2Target.b, finalModels: frames.at(-1).models}));
