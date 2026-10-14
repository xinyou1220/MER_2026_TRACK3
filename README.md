# MER 2026: From Discriminative Emotion Recognition to Generative Emotion Understanding — Track 3: MER-Prefer

## SymPO: Consistency by Construction via Permutation Symmetry for Preference Optimization

Meng-Ching Chen, Xin-You Song, Sheng-Bi Chen, Chia-Yen Lee (National Yang Ming Chiao Tung University)

*MRAC '26: 4th International Workshop on Multimodal, Generative and Responsible Affective Computing, ACM Multimedia 2026.* DOI: [10.1145/3840474.3840539](https://doi.org/10.1145/3840474.3840539)

This repository contains the code needed to reproduce our Track 3 results: data splits, SymPO
training, the baselines and ablations reported in the paper, held-out evaluation, and prediction of
the official test set.

## Task

MER-Prefer gives a short audio-visual clip and two candidate emotion analyses (`a1`, `a2`) written
by different systems. The model predicts which analysis human annotators preferred. Labels are
binary (`a1` / `a2`; `same` rows are excluded) and the metric is two-class weighted F1 (WAF-2).

A pairwise judge should give the same answer whichever candidate is shown first. Learned judges
often do not, and RL fine-tuning can turn a slot preference into a stable policy. SymPO enforces
this symmetry during training and at inference:

1. **Balanced swap rollouts.** Every pair appears in both orders in the same batch, and each order
   is sampled K = 4 times. With a +1 / −1 correctness reward, a policy that always picks the same
   slot earns an expected reward of exactly 0.
2. **Composite reward.** Correctness plus a label-free swap-consistency term (agreement with the
   majority choice in the opposite order), an out-of-fold Bradley–Terry text-ranker margin and a
   candidate-length penalty. The three auxiliary terms are squashed as τ·tanh(·) with τ < 1, so
   they never rank a wrong answer above a right one. Unparsable outputs get a fixed reward of −1.7.
   With K = 4 the opposite order can split 2–2; such ties resolve to `a2` (`>= 0.5` in
   `sympo_mer/reward.py`), which is the rule used for the paper's runs.
3. **REINFORCE++ update.** Group-mean baseline per (pair, order), one global advantage
   normalization over the batch, PPO clipping and a k2 KL loss. There is no critic. The reference
   policy is the base model with the LoRA adapter disabled.
4. **Antisymmetrized inference.** Constrained A/B scoring in both orders, averaged:
   p̂(a2 ≻ a1) = ½[q(a1, a2) + 1 − q(a2, a1)]. The decision is order-invariant by construction.
5. **Model selection** by this swap-averaged WAF-2 on a validation split drawn from the same weakly
   labeled source as the internal test split, never on the gold training subset.

Backbone: the Qwen2.5-Omni-7B thinker with LoRA (r = 16) on the attention and MLP projections.

## Results

Official Track 3 test set (379 pairs), WAF-2. Mean ± s.d. over three seeds where given.

| Method | WAF-2 |
|---|---|
| Qwen2.5-Omni zero-shot (organizer baseline) | 78.74 |
| One-shot prompting | 84.71 |
| SFT, no swap augmentation | 83.67 |
| SFT, swap augmentation | 90.51 ± 0.45 |
| Naive RL + antisymmetrized inference | 90.24 ± 1.47 |
| RL + balanced swap, correctness reward only | 89.89 ± 0.15 |
| SymPO-Additive (format bonus instead of the invalid floor) | 89.89 ± 1.25 |
| **SymPO** | **90.59 ± 1.19** |

Order dependence measured before swap averaging, on the internal held-out split (539 pairs):

| Method | Swap consistency ↑ | Position bias ↓ | Leakage (nats) ↓ |
|---|---|---|---|
| Zero-shot | 49.91 | 25.05 | 0.66 |
| SFT, swap augmentation | 83.36 ± 2.04 | 6.77 ± 1.86 | 0.44 ± 0.11 |
| Naive RL | 49.35 ± 17.58 | 25.08 ± 9.06 | 4.58 ± 2.24 |
| RL + balanced swap, correctness only | 65.99 ± 14.75 | 16.39 ± 7.65 | 3.06 ± 1.02 |
| SymPO-Additive | 58.38 ± 8.74 | 20.62 ± 4.47 | 4.93 ± 0.35 |
| **SymPO** | **79.28 ± 6.13** | **6.65 ± 6.46** | **1.96 ± 0.80** |

Naive RL reaches a similar test score, but its rollouts collapse to a single slot in all three
seeds. The swap-averaged decision hides this, so we report the symmetry metrics separately.

## Repository layout

```
sympo_mer/
  data.py          load the official CSVs and JSONL manifests, resolve media
  bt.py            out-of-fold Bradley-Terry text ranker (bt_a, bt_b)
  rollout.py       balanced-swap batch layout and rollouts
  reward.py        correctness, swap consistency, BT, length, bounded envelope, invalid floor
  advantage.py     group-mean baseline, global (or per-group) normalization
  omni_policy.py   Qwen2.5-Omni + LoRA: prompt, sampling, log-probs, constrained A/B scoring
  trainer.py       REINFORCE++ (PPO clip + k2 KL) / SFT training loop, checkpoint selection
  metrics.py       WAF-2 and the swap-averaged selection score
  scoring.py       both-order scoring, antisymmetrized decision, symmetry diagnostics
  config.py        default hyperparameters (paper values)
  experiments.py   settings for every row of the paper tables
scripts/
  split_weak_data.py    step 1: split EmoPrefer-V2 and build the combined training list
  build_splits.py       step 2: train / validation / held-out test splits + OOF-BT utilities
  train.py              step 3: train one model
  evaluate.py           step 4: evaluate on the held-out test split
  infer_official.py     step 5: predict the official test set
  package_submission.py zip the submission
  run_experiment.py     train + evaluate one named experiment
  summarize.py          mean ± s.d. table over seeds
run.sh                  one full run: splits, training, evaluation, submission
tests/test_core.py      CPU tests (no model needed)
```

## Setup

Python 3.11 and one CUDA GPU. The paper runs used one NVIDIA H100 (80 GB) per job.

```bash
conda create -n sympo python=3.11 -y && conda activate sympo
conda install -c conda-forge ffmpeg -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python tests/test_core.py
```

`transformers` is pinned to the Qwen2.5-Omni preview release used for the paper. Newer versions
place the LoRA adapters differently (the vision-encoder MLP is no longer matched), so checkpoints
are not interchangeable between versions and results can differ.

The base model `Qwen/Qwen2.5-Omni-7B` is downloaded from Hugging Face on first use. Pass a local
directory with `--model` if you have a copy.

## Data

The MER 2026 data is distributed by the organizers under their own license and is not included
here. Request access through the [challenge dataset page](https://huggingface.co/datasets/MERChallenge/MER2026)
and place the Track 3 files as follows:

```
data/        track3_emoprefer.csv    + video/ audio/ transcripts/   (EmoPrefer-Data, majority-voted)
data_weak/   track3_emopreferv2.csv  + video/ audio/ transcripts/   (EmoPrefer-Data-V2, single annotator)
data_test/   track3_candidate.csv    + video/ audio/ transcripts/   (official test, unlabeled)
```

Each sample `<name>` needs `video/<name>.mp4`, `audio/<name>.wav` and `transcripts/<name>.json`.
The transcript JSON must contain a `text` field (or `segments` with `text`); it is used as the ASR
transcript in the prompt.

## Reproducing the results

**1. Split the weakly labeled list and build the combined training list.**

```bash
python scripts/split_weak_data.py
```

Writes `data_weak/track3_emopreferv2_train.csv` (698 rows), `data_weak/track3_emopreferv2_val.csv`
(1,398 rows) and `data_mixed/train_gold_weak.jsonl` (1,272 rows).

**2. Build the splits and the out-of-fold BT utilities.**

```bash
python scripts/build_splits.py
```

Expected sizes, all binary: train 1,097, validation 539, held-out test 539, gold selection 557.
Six sample IDs present in both the gold list and the weak validation list are removed everywhere.
Sizes, label counts and SHA-256 hashes are written to `splits/manifest.json`.

**3. Train and evaluate SymPO.**

```bash
python scripts/run_experiment.py --experiment sympo --seed 42
```

This trains for 100 steps (4 pairs × 2 orders × 4 samples per step, learning rate 5e-5), selects
the checkpoint with the best swap-averaged WAF-2 on the validation split (evaluated every 25 steps),
and evaluates it on the held-out test split. The checkpoint is written to
`runs/sympo/seed42/best_trainable.pt` (LoRA weights only) and the metrics to
`results/sympo/seed42/test_metrics.json`. Repeat with other `--seed` values for the mean ± s.d.

**4. Predict the official test set.**

```bash
python scripts/infer_official.py --checkpoint runs/sympo/seed42/best_trainable.pt \
    --test data_test/track3_candidate.csv --out outputs/submission.csv
python scripts/package_submission.py --input outputs/submission.csv --out outputs/submission.zip
```

`submission.csv` has the columns `name,preference` with values `a1` / `a2`.

**5. Baselines and ablations.** Each row of the paper tables is a named experiment
(`python scripts/run_experiment.py --list` prints the settings):

| Paper row | `--experiment` |
|---|---|
| SymPO | `sympo` |
| Zero-shot, single order / + swap averaging | `zero_shot_single`, `zero_shot_swap` |
| SFT without / with swap augmentation | `sft_plain`, `sft_swap` |
| Naive RL + antisymmetrized inference | `naive_rl` |
| RL + balanced swap, correctness only | `rl_swap_correctness` |
| SymPO-Additive | `sympo_additive` |
| Remove balanced swap | `no_balanced_swap` |
| Per-group normalization | `per_group_norm` |
| Remove consistency reward | `no_consistency_reward` |
| Unbounded auxiliary | `unbounded_aux` |
| Continuous consistency | `continuous_consistency` |
| Remove BT margin / length debiasing | `no_bt`, `no_length` |
| Gold-subset selection | `gold_selection` |

The one-shot prompting baseline uses the base model on the official test set with one labeled
example from the gold list:

```bash
python scripts/infer_official.py --one_shot_source data/track3_emoprefer.csv \
    --inference_mode single_order --out outputs/one_shot.csv
```

After running several experiments and seeds, `python scripts/summarize.py` writes
`results/summary.md`.

`run.sh` runs steps 1–4 for one experiment and seed:

```bash
EXPERIMENT=sympo SEED=42 bash run.sh
```

## Notes on reproducibility

- Media settings used throughout: video at 2 fps, at most 8 frames, 3,136–50,176 pixels per frame,
  the separate wav track as audio, bf16 weights, SDPA attention. They are the defaults of every
  script.
- Sampling during training and GPU kernels are not fully deterministic across hardware and
  library versions. Expect small differences between runs with the same seed on different setups.
  The paper reports three seeds for this reason.
- The Qwen3-Omni-30B-A3B result and the label-free continuation experiment in the paper are not
  part of this release.

## Citation

```bibtex
@inproceedings{chen2026sympo,
  title     = {SymPO: Consistency by Construction via Permutation Symmetry for Preference Optimization},
  author    = {Chen, Meng-Ching and Song, Xin-You and Chen, Sheng-Bi and Lee, Chia-Yen},
  booktitle = {Proceedings of the 4th International Workshop on Multimodal, Generative and Responsible Affective Computing (MRAC '26)},
  year      = {2026},
  doi       = {10.1145/3840474.3840539}
}
```

The optimizer follows REINFORCE++ (Hu et al., arXiv:2501.03262). The task and data are described in
MER 2026 (Lian et al., arXiv:2604.19417).

## License

The code is released under the MIT License. The MER 2026 data and the Qwen2.5-Omni weights are
subject to their own licenses.
