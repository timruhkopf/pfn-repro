# Ported from pfn-acqpolicy@980fad9 src/pfn_acqpolicy/trainer/pfn_trainer.py, unchanged apart from import paths
# and the prior's / model's type hints (any prior with the episode interface, any surrogate).
"""PFN pretraining loop: a fresh BNN-prior batch every step, bar-distribution NLL,
AdamW + linear-warmup cosine schedule (PFNs4BO's recipe).

Carried over from AnytimeAcquisition; changed here for day-long GPU runs:
- `mixed_precision` is bf16 autocast (no GradScaler needed; the fp16 path was
  never validated on a GPU). Only the model forward runs under it: data and
  targets (`sample_batch`) and the loss stay fp32.
- `sample_batch` / `forward` are the hooks the alpha-PFN stages override
  (`trainer/es_trainer.py`).
- `grad_clip`, and `save_every` + `resume`: the checkpoint at `checkpoint_path`
  is rewritten periodically with optimizer/scheduler/RNG state, so a crashed
  run continues from its last save instead of from step 0.
- `deadline` (ISO time or epoch seconds): the GPU is booked until then. With
  `fit_to_deadline`, the step rate is measured over the first `calib_steps` steps and
  `n_steps` (hence the cosine schedule) shrinks so the run ends at about `deadline_margin`
  of the remaining time. In any case the run saves and stops once the deadline passes,
  so the last checkpoint is always usable.

The checkpoint is `{model_state, config, history, step, train_state, **extra}`;
`models/checkpoint.py::load_pfn_checkpoint` only reads `model_state`/`config`.
"""
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch

from pfn_repro.training.callbacks import Callback, CallbackHandler
from pfn_repro.surrogates.pfn import PFN
from pfn_repro.priors.bnn import BNNPrior


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def parse_deadline(v) -> float | None:
    """None, epoch seconds, or an ISO time ('2026-10-21T05:45') in local time -> epoch seconds."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return datetime.fromisoformat(str(v)).timestamp()


class PFNTrainer:
    def __init__(
        self,
        prior: Any,  # BNNPrior or MFPrior: the episode interface in priors/__init__.py
        model: torch.nn.Module,  # any surrogate: forward(x_train, y_train, x_test, n_features) + .bar_dist
        seed: int = 0,
        n_steps: int = 500,
        min_train: int = 3,
        max_train: int = 100,
        n_test: int = 1000,
        lr: float = 1e-3,
        warmup_steps: int = 50,
        grad_clip: float | None = None,
        log_every: int = 50,
        save_every: int | None = None,
        resume: bool = False,
        checkpoint_path: str | Path | None = None,
        model_config: dict | None = None,
        on_log: Callable[[int, dict], None] | None = None,
        mixed_precision: bool = False,
        extra_checkpoint_metadata: dict | None = None,  # lineage, e.g. {mlflow_run_id, git_commit}
        callbacks: list[Callback] | None = None,
        deadline: str | float | None = None,
        fit_to_deadline: bool = False,
        calib_steps: int = 300,
        deadline_margin: float = 0.9,
    ):
        self.prior = prior
        self.model = model
        self.bar_dist = model.bar_dist
        self.seed = seed
        self.n_steps = n_steps
        self.min_train = min_train
        self.max_train = max_train
        self.n_test = n_test
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.save_every = save_every
        self.resume = resume
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.model_config = model_config
        self.on_log = on_log
        self.mixed_precision = mixed_precision
        self.extra_checkpoint_metadata = extra_checkpoint_metadata or {}
        self.callback_handler = CallbackHandler(callbacks)
        self.deadline = parse_deadline(deadline)
        self.fit_to_deadline = fit_to_deadline
        self.calib_steps = calib_steps
        self.deadline_margin = deadline_margin
        self.stopped_at_deadline = False

    def sample_batch(self, step: int, n_train: int, generator: torch.Generator) -> tuple[dict, torch.Tensor]:
        """-> (model inputs, target per test point). Runs outside autocast."""
        self.prior.reset()
        x_tr, y_tr, x_te, y_te = self.prior.sample_episode(n_train, self.n_test)
        return dict(x_train=x_tr, y_train=y_tr, x_test=x_te, n_features=self.prior.active_dim), y_te

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(**batch)

    def _save(self, step: int, history: dict, optimizer, scheduler, generator) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_suffix(".tmp")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.model_config,
            "history": history,
            "step": step,
            "train_state": {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "generator": generator.get_state(),
                "prior_generator": self.prior.generator.get_state(),
                "n_steps": self.n_steps,
            },
            **self.extra_checkpoint_metadata,
        }, tmp)
        tmp.replace(self.checkpoint_path)  # atomic: a crash mid-save keeps the previous checkpoint

    def run(self) -> dict:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: cosine_warmup_lr(step, self.warmup_steps, self.n_steps)  # reads n_steps live
        )
        device_type = next(self.model.parameters()).device.type
        use_amp = self.mixed_precision and device_type == "cuda"
        generator = torch.Generator().manual_seed(self.seed)
        history = {"step": []}
        start = 0

        if self.resume and self.checkpoint_path is not None and self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=device_type, weights_only=False)
            if ckpt.get("step", -1) >= self.n_steps - 1:
                raise RuntimeError(f"{self.checkpoint_path} is already a finished run (step {ckpt['step']})")
            self.model.load_state_dict(ckpt["model_state"])
            ts = ckpt["train_state"]
            self.n_steps = min(self.n_steps, ts.get("n_steps", self.n_steps))  # keep a fitted schedule
            optimizer.load_state_dict(ts["optimizer"])
            scheduler.load_state_dict(ts["scheduler"])
            generator.set_state(ts["generator"])
            self.prior.generator.set_state(ts["prior_generator"])
            history, start = ckpt["history"], ckpt["step"] + 1
            print(f"resumed from {self.checkpoint_path} at step {start}")

        t0 = time.perf_counter()
        t_calib = None
        step = start
        while step < self.n_steps:
            if self.deadline is not None:
                if time.time() >= self.deadline:
                    self.stopped_at_deadline = True
                    print(f"deadline reached at step {step}: saving and stopping", flush=True)
                    break
                if self.fit_to_deadline and step == start + 20:
                    t_calib = time.perf_counter()        # after warm-up / first validation
                if self.fit_to_deadline and t_calib is not None and step == start + 20 + self.calib_steps:
                    rate = (time.perf_counter() - t_calib) / self.calib_steps
                    fit = step + int(self.deadline_margin * (self.deadline - time.time()) / rate)
                    self.n_steps = max(step + 1, min(self.n_steps, fit))
                    print(f"{rate * 1e3:.0f} ms/step: n_steps set to {self.n_steps}", flush=True)
                    if self.on_log is not None:
                        self.on_log(step, {"n_steps_fitted": float(self.n_steps), "sec_per_step": rate})
            n_train = int(torch.randint(self.min_train, self.max_train + 1, (1,), generator=generator).item())
            batch, y_te = self.sample_batch(step, n_train, generator)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
                logits = self.forward(batch)
            loss = self.bar_dist(logits.float(), y_te).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = (torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
                         if self.grad_clip else float("nan"))
            optimizer.step()
            scheduler.step()

            if step % self.log_every == 0 or step == self.n_steps - 1:
                with torch.no_grad():
                    mse = (self.bar_dist.mean(logits.float()) - y_te).square().mean().item()
                metrics = {"nll/train": loss.item(), "mse/train": mse, "grad_norm": grad_norm,
                           "lr": scheduler.get_last_lr()[0]}
                self.model.eval()
                metrics.update(self.callback_handler.run(step, self, self.log_every))
                self.model.train()
                history["step"].append(step)
                for k, v in metrics.items():
                    history.setdefault(k, []).append(v)
                print(f"step {step:7d}  nll={loss.item():.4f}  mse={mse:.4f}  n_train={n_train}  "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
                if self.on_log is not None:
                    self.on_log(step, metrics)

            if self.checkpoint_path is not None and self.save_every and step % self.save_every == 0 and step > start:
                self._save(step, history, optimizer, scheduler, generator)
            step += 1

        if self.checkpoint_path is not None:
            last = step - 1 if self.stopped_at_deadline else self.n_steps - 1
            if self.stopped_at_deadline:
                self.n_steps = step  # the checkpoint then reads as finished at its last step
            self._save(last, history, optimizer, scheduler, generator)
            print("saved checkpoint to", self.checkpoint_path)
        return {"model": self.model, "bar_dist": self.bar_dist, "prior": self.prior, "history": history}
