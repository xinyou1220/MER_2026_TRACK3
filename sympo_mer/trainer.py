"""Training loop: sampled REINFORCE++ (SymPO and RL baselines) or cross-entropy (SFT baselines)."""

from __future__ import annotations

import dataclasses
import json
import os
import random

import torch
from tqdm.auto import tqdm

from .advantage import advantage_reinforce_baseline, masked_mean
from .metrics import evaluate_waf2_swap
from .rollout import build_rollout, make_slot_batch


class Trainer:
    def __init__(self, policy, train_pairs, eval_pairs, cfg):
        self.policy = policy
        self.train_pairs = list(train_pairs)
        self.eval_pairs = eval_pairs
        self.cfg = cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.train.lr)
        self.rng = random.Random(cfg.train.seed)
        self.best_metric = -1.0
        self.topk = []

    def _optimize(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
        self.opt.step()

    def _ppo_loss(self, rollout, adv, old_logp):
        mask = rollout.action_mask
        eps = self.cfg.train.clip_eps
        new_logp = self.policy.logprobs(rollout)
        ratio = torch.exp((new_logp - old_logp) * mask)
        pg_loss = masked_mean(-torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv), mask)
        ref_logp = self.policy.ref_logprobs(rollout)
        kl_loss = torch.tensor(0.0, device=mask.device)
        if self.cfg.kl_loss_coef > 0:
            # k2 estimator: its gradient is the reverse-KL gradient.
            kl_loss = masked_mean(0.5 * (new_logp - ref_logp).pow(2), mask)
        return pg_loss + self.cfg.kl_loss_coef * kl_loss, kl_loss.item()

    def _rl_step(self, pairs):
        rd = build_rollout(self.policy, pairs, self.cfg, self.cfg.device)
        adv, _, _ = advantage_reinforce_baseline(rd.reward, rd.group_id, rd.rollout.action_mask,
                                                 self.cfg.adv.norm_eps, self.cfg.adv.normalization_mode)
        loss, kl = self._ppo_loss(rd.rollout, adv.detach(), rd.rollout.old_logp.detach())
        self._optimize(loss)
        labeled = rd.info.has_label
        return {
            "loss": loss.item(),
            "kl": kl,
            "reward": rd.reward.mean().item(),
            "acc": rd.info.correct[labeled].float().mean().item() if labeled.any() else 0.0,
            "slot1_rate": rd.info.slot1_rate,
            "consistency": rd.info.consistency.mean().item(),
        }

    def _sft_step(self, pairs):
        orientations = [0, 1] if self.cfg.rollout.balanced_swap else [0]
        slot_batch, meta = make_slot_batch(pairs, orientations, group_k=1, device=self.cfg.device)
        logp_a, logp_b = self.policy.ab_logprobs_train(slot_batch)
        # Slot A holds the preferred candidate when gold == orientation.
        target_a = (meta["gold"] == meta["orientation"]).float().to(logp_a.device)
        loss = -(target_a * logp_a + (1.0 - target_a) * logp_b).mean()
        self._optimize(loss)
        p_a = logp_a.detach().exp()
        return {"loss": loss.item(), "acc": (target_a * p_a + (1 - target_a) * (1 - p_a)).mean().item(),
                "slot1_rate": p_a.mean().item()}

    def _save(self, directory, step, metric, name="best_trainable"):
        meta = {"step": step, "metric": metric, **dataclasses.asdict(self.cfg)}
        self.policy.save_checkpoint(directory, meta, f"{name}.pt", f"{name.replace('trainable', 'meta')}.json")

    def _evaluate(self, step):
        model = self.policy.model
        model.eval()
        metric = evaluate_waf2_swap(self.policy, self.eval_pairs, self.cfg.device, self.cfg.train.eval_batch_size)
        model.train()

        out = self.cfg.train.output_dir
        k = self.cfg.train.topk_checkpoints
        if k > 0 and (len(self.topk) < k or metric > min(e[0] for e in self.topk)):
            name = f"top_step{step}_metric{metric:.4f}"
            self._save(os.path.join(out, "topk"), step, metric, name)
            self.topk = sorted(self.topk + [(metric, name)], reverse=True)
            for _, dropped in self.topk[k:]:
                for suffix in (".pt", ".json"):
                    os.remove(os.path.join(out, "topk", dropped + suffix))
            self.topk = self.topk[:k]
        if metric > self.best_metric:
            self.best_metric = metric
            self._save(out, step, metric)
        return metric

    def train(self):
        out = self.cfg.train.output_dir
        os.makedirs(out, exist_ok=True)
        for step in tqdm(range(1, self.cfg.train.steps + 1), dynamic_ncols=True, desc="train"):
            pairs = self.rng.sample(self.train_pairs, min(self.cfg.rollout.prompts_per_batch, len(self.train_pairs)))
            log = self._sft_step(pairs) if self.cfg.adv.estimator == "sft" else self._rl_step(pairs)
            log["step"] = step
            if step % self.cfg.train.eval_every == 0:
                log["waf2_swap"] = self._evaluate(step)
            with open(os.path.join(out, "train_metrics.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log) + "\n")
            if step % self.cfg.train.log_every == 0 or "waf2_swap" in log:
                print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in log.items()), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self.best_metric
