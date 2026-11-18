import unittest
import subprocess
import json
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import torch
from debugger import Config, Debugger


def equal(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a:
            equal(test, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b, strict=True):
            equal(test, x, y)
    else:
        test.assertEqual(a, b)

class Tests(unittest.TestCase):
    def test_every_phase_restore_is_bitwise_identical(self):
        baseline = Debugger()
        checkpoints = []
        while baseline.phase != 'done':
            checkpoints.append(baseline.snapshot())
            baseline.next()
        expected = baseline.snapshot()
        for checkpoint in checkpoints:
            with self.subTest(phase=checkpoint['phase'], update=checkpoint['updates']):
                restored = Debugger()
                restored.restore(checkpoint)
                restored.run()
                equal(self, expected, restored.snapshot())

    def test_back_and_rerun_each_step(self):
        d = Debugger()
        while d.phase != 'done':
            d.next()
            expected = d.snapshot()
            d.back()
            d.next()
            equal(self, expected, d.snapshot())

    def test_gradient_boundaries(self):
        d = Debugger()
        while d.phase != 'actor_backward':
            d.next()
        before = deepcopy(d.models['actor'].state_dict())
        critic = deepcopy(d.models['q1'].state_dict())
        d.next()
        equal(self, before, d.models['actor'].state_dict())
        equal(self, critic, d.models['q1'].state_dict())
        self.assertEqual(d.observed['q1_parameter_grads'], [None, None])
        for slope, grad in zip(d.observed['d_q_da'], d.observed['d_loss_da'], strict=True):
            self.assertEqual(slope[0], -2*grad[0])
        d.next()
        equal(self, critic, d.models['q1'].state_dict())
        self.assertFalse(torch.equal(before['w'], d.models['actor'].w))

    def test_critic_contribution_sum(self):
        d = Debugger()
        d.next(); d.next(); d.next()
        total = sum(x[0] for x in d.observed['q1_action_weight_contributions'])
        self.assertAlmostEqual(total, d.models['q1'].w.grad[2].item(), places=15)

    def test_terminal_fork_preserves_replay(self):
        original = Debugger()
        prior = original.snapshot()
        fork = original.fork(terminal_override=[1, 1])
        fork.next()
        equal(self, fork.y, fork.batch['r'])
        equal(self, original.snapshot(), prior)
        equal(self, fork.replay, original.replay)

    def test_config_fork_and_deep_snapshot(self):
        d = Debugger()
        saved = d.snapshot()
        fork = d.fork(config=replace(d.config, gamma=0.5))
        fork.models['actor'].w.data.add_(1)
        equal(self, saved, d.snapshot())
        d.run()
        d.restore(saved)
        equal(self, saved, d.snapshot())
        self.assertEqual(d.actor_opt.state_dict()['state'], {})
        with self.assertRaises(ValueError):
            fork.restore(saved)

    def test_finite_difference_actor_parameter(self):
        d = Debugger()
        while d.phase != 'actor_backward':
            d.next()
        d.next()
        p = d.models['actor'].w
        analytic = p.grad[0].item()
        x, eps = p[0].item(), 1e-6
        with torch.no_grad():
            p[0] = x + eps
            upper = d._actor_forward().item()
            p[0] = x - eps
            lower = d._actor_forward().item()
            p[0] = x
        self.assertAlmostEqual(analytic, (upper-lower)/(2*eps), places=9)

    def test_browser_and_torch_agree(self):
        result = subprocess.run(['node', str(Path(__file__).with_name('check_browser.cjs'))],
                                check=True, capture_output=True, text=True)
        browser = json.loads(result.stdout)
        d = Debugger()
        d.next()
        self.assertAlmostEqual(browser['target'], d.y[0].item(), places=14)
        d.run()
        self.assertAlmostEqual(browser['q2TargetBias'], d.models['q2_target'].b.item(), places=14)
        for name, model in d.models.items():
            js_name = name.replace('_target', 'Target')
            expected = browser['finalModels'][js_name]
            self.assertTrue(torch.allclose(model.w, torch.tensor(expected['w'], dtype=model.w.dtype), rtol=0, atol=1e-14))
            self.assertAlmostEqual(model.b.item(), expected['b'], places=14)

    def test_config_validation(self):
        for kwargs in ({'delay': 0}, {'gamma': float('nan')}, {'actor_lr': -1}, {'tau': 2}):
            with self.assertRaises(ValueError):
                Config(**kwargs)

if __name__ == '__main__':
    unittest.main()
