"""CPU/float64 TD3 teaching executor; no environment training or Adam.

Run: python debugger.py. Each next() performs exactly one phase. Snapshots
are trusted in-process objects, never serialized executable pickle inputs.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import torch
from torch import nn

DTYPE = torch.float64

@dataclass(frozen=True)
class Config:
    gamma: float = 0.9
    tau: float = 0.1
    critic_lr: float = 0.1
    actor_lr: float = 0.12
    delay: int = 2
    noise_std: float = 1.0
    noise_clip: float = 0.2

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Configuration must be finite")
        if not 0 <= self.gamma <= 1 or not 0 <= self.tau <= 1:
            raise ValueError("gamma/tau must lie in [0, 1]")
        if isinstance(self.delay, bool) or not isinstance(self.delay, int) or self.delay < 1:
            raise ValueError("delay must be a positive integer")
        if min(self.critic_lr, self.actor_lr, self.noise_std, self.noise_clip) < 0:
            raise ValueError("Learning rates and noise scales must be nonnegative")

class Model(nn.Module):
    def __init__(self, weights, bias, actor=False):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(weights, dtype=DTYPE))
        self.b = nn.Parameter(torch.tensor(bias, dtype=DTYPE))
        self.actor = actor

    def forward(self, s, a=None):
        x = s if self.actor else torch.cat((s, a, a.square()), dim=1)
        y = x @ self.w[:, None] + self.b
        return y.tanh() if self.actor else y

class Debugger:
    """Fixed replay, exact phase cursor, deep snapshots and deterministic restore.

    Config changes fork a fresh executor from initialization. This conservative
    boundary invalidates all derived training state, never the original replay.
    """
    def __init__(self, config=Config()):
        self.config = config
        self.models = {
            "actor": Model([0.2, -0.1], 0.1, True),
            "q1": Model([0.1, -0.05, 0.6, -0.4], 0.1),
            "q2": Model([0.08, 0.02, 0.4, -0.35], 0.02),
        }
        self.models.update({k + '_target': deepcopy(v) for k, v in list(self.models.items())})
        for name, model in self.models.items():
            if name.endswith('_target'):
                model.requires_grad_(False)
        self.actor_opt = torch.optim.SGD(self.models['actor'].parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.SGD(
            list(self.models['q1'].parameters()) + list(self.models['q2'].parameters()), lr=config.critic_lr)
        self.replay = {
            's': torch.tensor([[0.3, 0], [0.6, 0.3]], dtype=DTYPE),
            'a': torch.tensor([[0.6], [0.5]], dtype=DTYPE),
            'r': torch.tensor([[-0.1], [1]], dtype=DTYPE),
            'next': torch.tensor([[0.6, 0.3], [1, 0.4]], dtype=DTYPE),
            'd': torch.tensor([[0.], [1.]], dtype=DTYPE),
        }
        # Purpose-separated streams. The golden fixture intentionally uses fixed
        # base draws, so neither restore nor graph reconstruction consumes RNG.
        self.rng = {name: torch.Generator().manual_seed(seed)
                    for name, seed in [('replay', 11), ('exploration', 22), ('target', 33)]}
        self.base_z = torch.tensor([[0.3], [-0.3]], dtype=DTYPE)
        self.batch = deepcopy(self.replay)
        self.phase = 'target'
        self.updates = 0
        self.observed = {}
        self.y = None
        self.loss = None
        self.action = None
        self.history = []

    def snapshot(self):
        return deepcopy({
            'config': asdict(self.config), 'phase': self.phase, 'updates': self.updates,
            'models': {k: v.state_dict() for k, v in self.models.items()},
            'grads': {k: [None if p.grad is None else p.grad.detach().clone() for p in m.parameters()]
                      for k, m in self.models.items()},
            'requires_grad': {k: [p.requires_grad for p in m.parameters()] for k, m in self.models.items()},
            'optimizers': [self.actor_opt.state_dict(), self.critic_opt.state_dict()],
            'rng': {k: g.get_state() for k, g in self.rng.items()},
            'replay': self.replay, 'batch': self.batch, 'base_z': self.base_z,
            'y': self.y, 'observed': self.observed,
        })

    def restore(self, state):
        state = deepcopy(state)
        if state['config'] != asdict(self.config):
            raise ValueError('Restore requires the same configuration; fork instead')
        for name, model in self.models.items():
            model.load_state_dict(state['models'][name])
            for p, grad, requires in zip(model.parameters(), state['grads'][name], state['requires_grad'][name], strict=True):
                p.requires_grad_(requires)
                p.grad = grad
        for opt, saved in zip((self.actor_opt, self.critic_opt), state['optimizers'], strict=True):
            opt.load_state_dict(saved)
        for name, g in self.rng.items():
            g.set_state(state['rng'][name])
        for key in ('phase', 'updates', 'replay', 'batch', 'base_z', 'y', 'observed'):
            setattr(self, key, state[key])
        self.loss = self.action = None  # Rebuild lazily only if backward is next.

    def back(self):
        if not self.history:
            raise ValueError('Already at initialization')
        self.restore(self.history.pop())

    def fork(self, *, config=None, terminal_override=None):
        other = Debugger(self.config if config is None else config)
        other.replay = deepcopy(self.replay)
        other.batch = deepcopy(self.replay)
        other.base_z = self.base_z.clone()
        if terminal_override is not None:
            if len(terminal_override) != len(other.batch['d']) or any(x not in (0, 1) for x in terminal_override):
                raise ValueError('Expected one binary terminal flag per sample')
            other.batch['d'] = torch.tensor(terminal_override, dtype=DTYPE)[:, None]
        return other

    def _critic_forward(self):
        b = self.batch
        return sum((self.models[k](b['s'], b['a']) - self.y).square() for k in ('q1', 'q2')).mean()

    def _actor_forward(self):
        self.models['q1'].requires_grad_(False)
        self.action = self.models['actor'](self.batch['s'])
        return -self.models['q1'](self.batch['s'], self.action).mean()

    def next(self):
        if self.phase == 'done':
            raise StopIteration
        self.history.append(self.snapshot())
        c, b, m = self.config, self.batch, self.models
        executed = self.phase
        if executed == 'target':
            with torch.no_grad():
                noise = (c.noise_std * self.base_z).clamp(-c.noise_clip, c.noise_clip)
                a = (m['actor_target'](b['next']) + noise).clamp(-1, 1)
                self.y = b['r'] + c.gamma * (1 - b['d']) * torch.minimum(
                    m['q1_target'](b['next'], a), m['q2_target'](b['next'], a))
            self.phase = 'critic_forward'
        elif executed == 'critic_forward':
            self.critic_opt.zero_grad(set_to_none=True)
            self.loss = self._critic_forward()
            self.observed['critic_loss'] = self.loss.item()
            self.phase = 'critic_backward'
        elif executed == 'critic_backward':
            if self.loss is None:
                self.loss = self._critic_forward()
            self.loss.backward()
            q = m['q1'](b['s'], b['a']).detach()
            # Each row's contribution to the SAME shared action weight.
            contributions = 2 * (q - self.y) * b['a'] / len(b['s'])
            self.observed['q1_action_weight_contributions'] = contributions.tolist()
            self.loss = None
            self.phase = 'critic_step'
        elif executed == 'critic_step':
            self.critic_opt.step()
            self.updates += 1
            self.phase = 'actor_forward' if self.updates % c.delay == 0 else ('target' if self.updates < 2 else 'done')
        elif executed == 'actor_forward':
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            self.loss = self._actor_forward()
            self.observed['actor_before'] = self.action.detach().tolist()
            self.observed['actor_loss'] = self.loss.item()
            self.phase = 'actor_backward'
        elif executed == 'actor_backward':
            if self.loss is None:
                self.loss = self._actor_forward()
            self.action.retain_grad()
            self.loss.backward()
            self.observed['d_loss_da'] = self.action.grad.tolist()
            self.observed['d_q_da'] = (-len(b['s']) * self.action.grad).tolist()
            self.observed['q1_parameter_grads'] = [None if p.grad is None else p.grad.tolist() for p in m['q1'].parameters()]
            self.observed['actor_grads'] = [p.grad.tolist() for p in m['actor'].parameters()]
            self.loss = self.action = None
            self.phase = 'actor_step'
        elif executed == 'actor_step':
            self.actor_opt.step()
            m['q1'].requires_grad_(True)
            self.observed['actor_after'] = m['actor'](b['s']).detach().tolist()
            self.phase = 'sync'
        elif executed == 'sync':
            with torch.no_grad():
                for name in ('actor', 'q1', 'q2'):
                    for p, target in zip(m[name].parameters(), m[name + '_target'].parameters(), strict=True):
                        target.copy_((1-c.tau)*target + c.tau*p)
            self.phase = 'target' if self.updates < 2 else 'done'
        return executed

    def run(self):
        while self.phase != 'done':
            self.next()
        return self.observed

if __name__ == '__main__':
    print(json.dumps(Debugger().run(), indent=2))
