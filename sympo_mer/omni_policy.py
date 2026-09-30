"""Qwen2.5-Omni policy with LoRA on the thinker.

The reference policy is the same model with the LoRA adapter disabled. Prompts contain the video
frames, the separate wav track, the ASR transcript and the two candidate analyses, and ask for a
single letter A or B.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import re
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from .rollout import Rollout

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DECISION_RE = re.compile(
    r"^\s*(?:(?:answer|preference|choice)\s*(?::|=|-|is)?\s*)?\(?\s*([AB])\s*\)?"
    r"(?=\s|[).,;:!?，。；：！？]|$)",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)

USER_TEMPLATE = """You are doing a two-class preference task. Compare two candidate multimodal emotion analyses for the same clip. Use the video frames, the separate wav audio, the ASR transcript, and the candidate evidence.

You must reply with exactly one character: A or B.

ASR transcript:
{asr_text}

Candidate A:
{candidate_a}

Candidate B:
{candidate_b}

Which candidate is better supported by the clip? Reply with exactly one character: A or B."""

ONE_SHOT_TEMPLATE = """Here is one labeled training example for the same two-class preference task. This demonstration provides its transcript and candidate evidence; its source media are not part of the current test clip.

ASR transcript:
{asr_text}

Candidate A:
{candidate_a}

Candidate B:
{candidate_b}

Which candidate is better supported? Reply with exactly one character: A or B."""


def _pad_left(t: torch.Tensor, width: int, value) -> torch.Tensor:
    pad = width - t.shape[1]
    if pad <= 0:
        return t
    return torch.cat([torch.full((t.shape[0], pad), value, dtype=t.dtype, device=t.device), t], dim=1)


class OmniPolicy:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Omni-7B",
        lora: bool = True,
        device: str = "cuda",
        max_new_tokens: int = 8,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        checkpoint_path: str | None = None,
        gradient_checkpointing: bool = True,
        video_fps: float = 2.0,
        video_min_pixels: int = 3136,
        video_max_pixels: int = 50176,
        video_max_frames: int = 8,
        chunk_size: int = 1,
        one_shot_example: dict | None = None,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.video_options = {"fps": video_fps, "min_pixels": video_min_pixels,
                              "max_pixels": video_max_pixels, "max_frames": video_max_frames}
        self.chunk_size = chunk_size
        self.one_shot_example = one_shot_example

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", attn_implementation="sdpa")
        self.model.disable_talker()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        if lora:
            thinker = self.model.thinker
            if gradient_checkpointing:
                thinker.gradient_checkpointing_enable()
            thinker.config.use_cache = False
            self.model.thinker = get_peft_model(thinker, LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                target_modules=LORA_TARGETS))
        self.model.train()

        params = inspect.signature(self._base_thinker().forward).parameters
        self._logits_keep_kw = next((k for k in ("logits_to_keep", "num_logits_to_keep") if k in params), None)

        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "left"
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.a_id, self.b_id = (tokenizer(x, add_special_tokens=False)["input_ids"][0] for x in ("A", "B"))
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def _base_thinker(self):
        thinker = self.model.thinker
        return thinker.get_base_model() if hasattr(thinker, "get_base_model") else thinker

    def _forward(self, inputs, keep: int, use_ref: bool = False):
        """Thinker forward that only projects the last `keep` positions through the LM head."""
        call = {**inputs, "use_cache": False}
        if self._logits_keep_kw:
            call[self._logits_keep_kw] = keep
        with self.model.thinker.disable_adapter() if use_ref else contextlib.nullcontext():
            logits = self.model.thinker(**call).logits
        return logits[:, -keep:, :]

    def _messages(self, video, audio, asr_text, cand_a, cand_b):
        content = []
        if video:
            content.append({"type": "video", "video": Path(video).resolve().as_posix(), **self.video_options})
        if audio:
            content.append({"type": "audio", "audio": Path(audio).resolve().as_posix()})
        content.append({"type": "text", "text": USER_TEMPLATE.format(
            asr_text=asr_text or "(empty)", candidate_a=cand_a or "(empty)", candidate_b=cand_b or "(empty)")})

        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        if self.one_shot_example:
            demo = self.one_shot_example
            demo_text = ONE_SHOT_TEMPLATE.format(asr_text=demo["asr_text"] or "(empty)",
                                                 candidate_a=demo["candidate_a"] or "(empty)",
                                                 candidate_b=demo["candidate_b"] or "(empty)")
            messages.append({"role": "user", "content": [{"type": "text", "text": demo_text}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": demo["answer"]}]})
        messages.append({"role": "user", "content": content})
        return messages

    def _build_inputs(self, slot_batch):
        conversations = [
            self._messages(*row) for row in zip(slot_batch["video_path"], slot_batch["audio_path"],
                                                slot_batch["asr_text"], slot_batch["candidate_slot1"],
                                                slot_batch["candidate_slot2"])
        ]
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True, use_audio_in_video=False)
        device = next(self.model.parameters()).device
        dtype = next(p.dtype for p in self.model.parameters() if p.is_floating_point())
        return {k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device))
                if torch.is_tensor(v) else v for k, v in inputs.items()}

    def _chunks(self, slot_batch):
        n = len(slot_batch["candidate_slot1"])
        size = max(1, min(self.chunk_size, n)) if self.chunk_size > 0 else max(n, 1)
        for start in range(0, n, size):
            yield {key: value[start:start + size] for key, value in slot_batch.items()}

    def ab_logprobs_train(self, slot_batch, use_ref: bool = False):
        """log P(A), log P(B) of the first answer token, renormalized over {A, B}."""
        scores_a, scores_b = [], []
        for chunk in self._chunks(slot_batch):
            inputs = self._build_inputs(chunk)
            with torch.no_grad() if use_ref else contextlib.nullcontext():
                logits = self._forward(inputs, keep=1, use_ref=use_ref)[:, -1, :]
            logits = logits.to(inputs["input_ids"].device).float()
            logp = torch.log_softmax(torch.stack([logits[:, self.a_id], logits[:, self.b_id]], dim=-1), dim=-1)
            scores_a.append(logp[:, 0])
            scores_b.append(logp[:, 1])
            torch.cuda.empty_cache()
        return torch.cat(scores_a), torch.cat(scores_b)

    @torch.no_grad()
    def score_ab_logprobs(self, slot_batch):
        return self.ab_logprobs_train(slot_batch)

    def _response_logp(self, inputs, action_mask, use_ref=False):
        """Teacher-forced log-probs of the response tokens, which occupy the trailing columns."""
        seq_ids = inputs["input_ids"]
        out = torch.zeros_like(action_mask, dtype=torch.float32)
        width = int(action_mask.sum(dim=1).max().item())
        if width == 0:
            return out
        logits = self._forward(inputs, keep=width + 1, use_ref=use_ref)[:, :-1, :].float().to(seq_ids.device)
        targets = seq_ids[:, -width:]
        out[:, -width:] = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        return out * action_mask

    def _generate_chunk(self, slot_batch):
        inputs = self._build_inputs(slot_batch)
        prompt_ids = inputs["input_ids"]
        width = prompt_ids.shape[1]
        kwargs = {"do_sample": self.do_sample, "max_new_tokens": self.max_new_tokens, "use_audio_in_video": False,
                  "return_audio": False, "pad_token_id": self.pad_token_id, "eos_token_id": self.eos_token_id}
        if self.do_sample:
            kwargs.update(temperature=self.temperature, top_p=self.top_p)
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(**inputs, **kwargs)
        self.model.train()
        text_ids = generated[0] if isinstance(generated, tuple) else generated

        if text_ids.shape[1] >= width and torch.equal(text_ids[:, :width], prompt_ids):
            seq_ids, response_ids = text_ids, text_ids[:, width:]
        else:
            response_ids = text_ids
            seq_ids = torch.cat([prompt_ids, response_ids.to(prompt_ids.device)], dim=1)
        attn = torch.cat([inputs["attention_mask"], torch.ones_like(response_ids, device=seq_ids.device)], dim=1)
        action_mask = torch.zeros_like(attn, dtype=torch.float)
        action_mask[:, -response_ids.shape[1]:] = 1.0

        full_inputs = {**inputs, "input_ids": seq_ids, "attention_mask": attn}
        with torch.no_grad():
            old_logp = self._response_logp(full_inputs, action_mask).detach()
        texts = self.processor.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {"inputs": full_inputs, "seq_ids": seq_ids, "attn": attn, "action_mask": action_mask,
                "old_logp": old_logp, "texts": texts}

    def generate(self, slot_batch) -> Rollout:
        """Sample one short answer per row; old_logp is teacher-forced under the current policy."""
        chunks = []
        for sub in self._chunks(slot_batch):
            chunks.append(self._generate_chunk(sub))
            torch.cuda.empty_cache()
        width = max(c["seq_ids"].shape[1] for c in chunks)
        return Rollout(
            seq_ids=torch.cat([_pad_left(c["seq_ids"], width, self.pad_token_id) for c in chunks]),
            attn=torch.cat([_pad_left(c["attn"], width, 0) for c in chunks]),
            action_mask=torch.cat([_pad_left(c["action_mask"], width, 0.0) for c in chunks]),
            old_logp=torch.cat([_pad_left(c["old_logp"], width, 0.0) for c in chunks]),
            aux={"chunks": chunks, "response_texts": [t for c in chunks for t in c["texts"]]},
        )

    def _rollout_logprobs(self, rollout: Rollout, use_ref: bool):
        width = rollout.seq_ids.shape[1]
        return torch.cat([_pad_left(self._response_logp(c["inputs"], c["action_mask"], use_ref), width, 0.0)
                          for c in rollout.aux["chunks"]])

    def logprobs(self, rollout: Rollout):
        return self._rollout_logprobs(rollout, use_ref=False)

    @torch.no_grad()
    def ref_logprobs(self, rollout: Rollout):
        return self._rollout_logprobs(rollout, use_ref=True)

    def parse_decision(self, rollout: Rollout):
        matches = [DECISION_RE.search(text.strip()) for text in rollout.aux["response_texts"]]
        decision = [(0 if m.group(1).upper() == "A" else 1) if m else -1 for m in matches]
        valid = [m is not None for m in matches]
        return torch.tensor(decision, device=self.device), torch.tensor(valid, device=self.device)

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_checkpoint(self, output_dir: str, metadata: dict, filename: str, meta_filename: str):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        state = {name: p.detach().cpu() for name, p in self.model.named_parameters() if p.requires_grad}
        torch.save(state, path / filename)
        (path / meta_filename).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load_checkpoint(self, checkpoint_path: str):
        own = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, value in torch.load(checkpoint_path, map_location="cpu").items():
                own[name].copy_(value.to(device=own[name].device, dtype=own[name].dtype))
