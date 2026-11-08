# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy>=1.26,<3", "torch>=2.4,<3"]
# ///
"""A1·S03 checkpoint 1: English character-level tiny GPT + single-batch overfit.

Wayne owns the new training-loop decisions: shifted data windows, embedding
broadcast, per-position vocabulary logits, flattened cross-entropy, and the
zero_grad -> forward -> backward -> step order. The attention/Pre-LN block is a
mechanical PyTorch port of the already-verified A1·S02 NumPy implementation.

Run from the repository root:
    uv run learners/wayne/code/a1/s03_tiny_gpt.py
    uv run learners/wayne/code/a1/s03_tiny_gpt.py --experiment objective-control
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


TRAIN_LINES = (
    "the small robot sees the red cube.\n",
    "the small robot moves to the red cube.\n",
    "the careful robot lifts the red cube.\n",
    "the small robot sees the blue bowl.\n",
    "the small robot moves to the blue bowl.\n",
    "the careful robot places the cube in the bowl.\n",
    "the camera sees the table and the robot.\n",
    "the policy reads the image and predicts an action.\n",
    "the action changes the next observation.\n",
    "the next observation changes the next action.\n",
)

VALIDATION_LINES = (
    "the careful robot sees the blue cube.\n",
    "the small robot lifts the blue bowl.\n",
    "the camera reads the next observation.\n",
    "the policy predicts the next action.\n",
)

# Split by sentence before repetition so no exact validation sentence leaks into train.
TRAINING_TEXT = "".join(TRAIN_LINES) * 32
VALIDATION_TEXT = "".join(VALIDATION_LINES) * 16


@dataclass(frozen=True)
class Config:
    block_size: int = 32
    batch_size: int = 8
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    learning_rate: float = 3e-3
    overfit_steps: int = 400
    comparison_steps: int = 800
    eval_batches: int = 20
    seed: int = 20260908


class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {char: index for index, char in enumerate(self.chars)}
        self.itos = {index: char for char, index in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[index] for index in ids)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_batch(
    data: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample B shifted windows and retain their corpus offsets for inspection."""
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device), starts


class CausalSelfAttention(nn.Module):
    """Mechanical PyTorch port of A1·S02 causal MHA."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, time, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :time, :time], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, time, width)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LN block; shape stays (B, T, D)."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, config: Config) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, time = idx.shape
        if time > self.config.block_size:
            raise ValueError(f"context length {time} exceeds block_size {self.config.block_size}")

        token_states = self.token_embedding(idx)
        positions = torch.arange(time, device=idx.device)
        position_states = self.position_embedding(positions)
        hidden = token_states + position_states

        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * time, self.vocab_size),
                targets.reshape(batch_size * time),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


@torch.no_grad()
def assert_causal_invariance(model: TinyGPT, x: torch.Tensor) -> None:
    """Changing a suffix must not change logits at earlier positions."""
    model.eval()
    probe_a = x[:1].clone()
    probe_b = probe_a.clone()
    cut = probe_a.shape[1] // 2
    probe_b[:, cut:] = (probe_b[:, cut:] + 1) % model.vocab_size
    logits_a, _ = model(probe_a)
    logits_b, _ = model(probe_b)
    if not torch.allclose(logits_a[:, :cut], logits_b[:, :cut], atol=1e-5, rtol=1e-5):
        max_diff = (logits_a[:, :cut] - logits_b[:, :cut]).abs().max().item()
        raise AssertionError(f"future tokens changed earlier logits; max diff={max_diff:.3e}")


def print_fixed_batch(
    tokenizer: CharTokenizer,
    fixed_x: torch.Tensor,
    fixed_y: torch.Tensor,
    starts: torch.Tensor,
    prompt: str,
) -> None:
    decoded_x = [tokenizer.decode(row.detach().cpu().tolist()) for row in fixed_x]
    decoded_y = [tokenizer.decode(row.detach().cpu().tolist()) for row in fixed_y]
    print("\n--- fixed windows sampled once, then reused for every optimizer step ---")
    for row, (start, x_text, y_text) in enumerate(zip(starts.tolist(), decoded_x, decoded_y)):
        print(f"row={row} corpus_start={start:5d} x={x_text!r}")
        print(f"{'':28s} y={y_text!r}")
    containing = [row for row, text in enumerate(decoded_x) if prompt in text]
    starting = [row for row, text in enumerate(decoded_x) if text.startswith(prompt)]
    print(f"\ngeneration prompt={prompt!r}")
    print(f"fixed rows containing the prompt: {containing}")
    print(f"fixed rows starting with the prompt at local position 0: {starting}")


@torch.no_grad()
def print_next_token_probe(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    *,
    label: str,
    prompt: str,
    expected_next: str,
    device: torch.device,
) -> None:
    """Report a deterministic next-token distribution for one prompt."""
    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    logits, _ = model(prompt_ids)
    probabilities = F.softmax(logits[0, -1], dim=-1)

    expected_id = tokenizer.stoi[expected_next]
    expected_probability = probabilities[expected_id].item()
    expected_rank = int((probabilities > probabilities[expected_id]).sum().item()) + 1
    top_probability, top_id = probabilities.max(dim=-1)
    top_token = tokenizer.itos[top_id.item()]

    print(
        f"{label}: prompt={prompt!r} positions=0..{len(prompt) - 1} | "
        f"top1={top_token!r} p_top1={top_probability.item():.4f} | "
        f"p_expected={expected_probability:.4f} rank_expected={expected_rank}"
    )


def objective_targets(kind: str, x: torch.Tensor, shifted_y: torch.Tensor) -> torch.Tensor:
    if kind == "next-token":
        return shifted_y
    if kind == "copy-current":
        return x
    raise ValueError(f"unknown objective kind: {kind}")


@torch.no_grad()
def evaluate_objective(
    model: TinyGPT,
    data: torch.Tensor,
    config: Config,
    device: torch.device,
    *,
    objective_kind: str,
    seed: int,
) -> tuple[float, float]:
    """Return mean loss and token accuracy on deterministic random windows."""
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for _ in range(config.eval_batches):
        x, shifted_y, _ = get_batch(
            data,
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=device,
            generator=generator,
        )
        targets = objective_targets(objective_kind, x, shifted_y)
        logits, loss = model(x, targets)
        assert loss is not None
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()
        total_tokens += targets.numel()
    if was_training:
        model.train()
    return total_loss / config.eval_batches, total_correct / total_tokens


def train_objective_model(
    tokenizer: CharTokenizer,
    train_data: torch.Tensor,
    config: Config,
    device: torch.device,
    *,
    label: str,
    objective_kind: str,
) -> TinyGPT:
    """Train one model; both comparison arms receive identical starts and initialization."""
    torch.manual_seed(config.seed)
    model = TinyGPT(tokenizer.vocab_size, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    batch_generator = torch.Generator().manual_seed(config.seed + 1)
    model.train()

    print(f"training {label}: objective={objective_kind}, steps={config.comparison_steps}")
    for step in range(1, config.comparison_steps + 1):
        x, shifted_y, _ = get_batch(
            train_data,
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=device,
            generator=batch_generator,
        )
        targets = objective_targets(objective_kind, x, shifted_y)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == config.comparison_steps:
            print(f"  step={step:4d} own_objective_loss={loss.item():.4f}")
    return model


@torch.no_grad()
def next_token_measurement(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    expected_next: str,
    device: torch.device,
) -> tuple[str, float]:
    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    logits, _ = model(prompt_ids)
    probabilities = F.softmax(logits[0, -1], dim=-1)
    expected_probability = probabilities[tokenizer.stoi[expected_next]].item()
    top_id = probabilities.argmax().item()
    return tokenizer.itos[top_id], expected_probability


def run_objective_comparison(config: Config) -> None:
    """Compare a real next-token baseline with an intentionally misaligned y=x arm."""
    device = choose_device()
    tokenizer = CharTokenizer(TRAINING_TEXT)
    unseen_characters = sorted(set(VALIDATION_TEXT) - set(TRAINING_TEXT))
    if unseen_characters:
        raise AssertionError(f"validation contains out-of-vocabulary characters: {unseen_characters}")
    train_data = torch.tensor(tokenizer.encode(TRAINING_TEXT), dtype=torch.long)
    validation_data = torch.tensor(tokenizer.encode(VALIDATION_TEXT), dtype=torch.long)

    print(
        f"device={device} train_chars={len(TRAINING_TEXT)} "
        f"validation_chars={len(VALIDATION_TEXT)} vocab={tokenizer.vocab_size}"
    )
    print("split check: PASS | validation sentences are held out before repetition")

    arms = (
        ("baseline", "next-token"),
        ("y=x control", "copy-current"),
    )
    trained: list[tuple[str, str, TinyGPT]] = []
    for label, objective_kind in arms:
        model = train_objective_model(
            tokenizer,
            train_data,
            config,
            device,
            label=label,
            objective_kind=objective_kind,
        )
        trained.append((label, objective_kind, model))

    probe_prompt = "the careful robot sees the blue"
    expected_next = " "
    sample_prompt = "the careful robot"
    rows: list[tuple[str, float, float, float, float, str, float]] = []
    samples: list[tuple[str, str]] = []
    for arm_index, (label, objective_kind, model) in enumerate(trained):
        own_train_loss, _ = evaluate_objective(
            model,
            train_data,
            config,
            device,
            objective_kind=objective_kind,
            seed=config.seed + 10,
        )
        true_train_loss, _ = evaluate_objective(
            model,
            train_data,
            config,
            device,
            objective_kind="next-token",
            seed=config.seed + 10,
        )
        true_val_loss, true_val_accuracy = evaluate_objective(
            model,
            validation_data,
            config,
            device,
            objective_kind="next-token",
            seed=config.seed + 20,
        )
        top_token, expected_probability = next_token_measurement(
            model,
            tokenizer,
            probe_prompt,
            expected_next,
            device,
        )
        rows.append(
            (
                label,
                own_train_loss,
                true_train_loss,
                true_val_loss,
                true_val_accuracy,
                top_token,
                expected_probability,
            )
        )

        prompt_ids = torch.tensor([tokenizer.encode(sample_prompt)], dtype=torch.long, device=device)
        torch.manual_seed(config.seed + arm_index)
        generated = model.generate(prompt_ids, max_new_tokens=80)
        samples.append((label, tokenizer.decode(generated[0].tolist())))

    print("\n--- unified task-contract evaluation ---")
    print(f"probe={probe_prompt!r}, expected_next={expected_next!r}")
    print(
        f"{'model':<13} {'own_train':>10} {'true_train':>11} "
        f"{'true_val':>9} {'val_acc':>8} {'top1':>7} {'p_expected':>11}"
    )
    for label, own_loss, true_train, true_val, val_acc, top_token, p_expected in rows:
        print(
            f"{label:<13} {own_loss:10.4f} {true_train:11.4f} "
            f"{true_val:9.4f} {val_acc:8.3f} {top_token!r:>7} {p_expected:11.4f}"
        )

    print("\n--- qualitative fixed-prompt samples (not a scored metric) ---")
    for label, sample in samples:
        print(f"{label}: {sample}")


def run_overfit(config: Config, *, inspect_only: bool = False) -> None:
    torch.manual_seed(config.seed)
    device = choose_device()
    tokenizer = CharTokenizer(TRAINING_TEXT)
    data = torch.tensor(tokenizer.encode(TRAINING_TEXT), dtype=torch.long)
    fixed_x, fixed_y, starts = get_batch(
        data,
        batch_size=config.batch_size,
        block_size=config.block_size,
        device=device,
    )

    bare_prompt = "the small robot"
    position_matched_prompt = ".\nthe small robot"
    expected_next = " "
    print_fixed_batch(tokenizer, fixed_x, fixed_y, starts, bare_prompt)
    if inspect_only:
        return

    model = TinyGPT(tokenizer.vocab_size, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    logits, initial_loss = model(fixed_x, fixed_y)
    assert initial_loss is not None
    assert logits.shape == (config.batch_size, config.block_size, tokenizer.vocab_size)
    assert_causal_invariance(model, fixed_x)
    before = model.token_embedding.weight.detach().clone()

    print(f"device={device} vocab={tokenizer.vocab_size} parameters={parameter_count:,}")
    print(f"fixed batch x/y shape={tuple(fixed_x.shape)} initial_loss={initial_loss.item():.4f}")
    print("shape check: PASS | causal invariance: PASS")

    model.train()
    loss = initial_loss
    for step in range(1, config.overfit_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(fixed_x, fixed_y)
        assert loss is not None
        loss.backward()
        if step == 1:
            grad = model.token_embedding.weight.grad
            if grad is None or not torch.isfinite(grad).all() or grad.abs().max().item() == 0:
                raise AssertionError("token embedding gradient is missing, non-finite, or zero")
        optimizer.step()
        if step % 50 == 0 or step == config.overfit_steps:
            print(f"step={step:4d} loss={loss.item():.4f}")

    delta = (model.token_embedding.weight.detach() - before).abs().max().item()
    if delta == 0:
        raise AssertionError("parameter delta is zero after optimizer steps")
    print(f"parameter delta check: PASS | token_embedding max_abs_delta={delta:.3e}")

    print("\n--- deterministic training-context A/B next-token probe ---")
    print(f"expected next token={expected_next!r}")
    print_next_token_probe(
        model,
        tokenizer,
        label="A bare",
        prompt=bare_prompt,
        expected_next=expected_next,
        device=device,
    )
    print_next_token_probe(
        model,
        tokenizer,
        label="B position-matched",
        prompt=position_matched_prompt,
        expected_next=expected_next,
        device=device,
    )

    prompt_ids = torch.tensor([tokenizer.encode(bare_prompt)], dtype=torch.long, device=device)
    torch.manual_seed(config.seed)
    generated = model.generate(prompt_ids, max_new_tokens=120)
    print("\n--- fixed-seed sample from A bare prompt ---")
    print(tokenizer.decode(generated[0].tolist()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("overfit", "objective-control"),
        default="overfit",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.experiment == "overfit":
        steps = Config.overfit_steps if args.steps is None else args.steps
        run_overfit(Config(overfit_steps=steps), inspect_only=args.inspect_only)
    else:
        if args.inspect_only:
            raise SystemExit("--inspect-only is available only for --experiment overfit")
        steps = Config.comparison_steps if args.steps is None else args.steps
        run_objective_comparison(Config(comparison_steps=steps))
