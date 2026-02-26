"""High-level orchestration for multi-part experiments."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from statistics import mean

import torch
import yaml

from .datasets import (
    DatasetBundle,
    TieredDatasetSuite,
    load_tiered_suite,
)
from .evaluation import BaselineEvaluator, run_parameter_sweep
from .hf_hooks import apply_hooks
from .hooks import AttentionHookConfig, NeuronHookConfig
from .model_loader import load_local_model
from .statistics import summarize


NUMBER_PATTERN = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
FRACTION_PATTERN = re.compile(
    r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+))\s*/\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+))"
)


def _coerce_numeric(text: str) -> Optional[int | float]:
    cleaned = text.strip().replace(",", "").replace("−", "-")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def _extract_numeric_from_text(text: str) -> Optional[int | float]:
    fraction_match = FRACTION_PATTERN.search(text)
    if fraction_match:
        numerator = _coerce_numeric(fraction_match.group(1))
        denominator = _coerce_numeric(fraction_match.group(2))
        if numerator is not None and denominator not in (None, 0):
            try:
                value = float(numerator) / float(denominator)
            except ZeroDivisionError:
                value = None
            if value is not None:
                if abs(value - round(value)) < 1e-9:
                    return int(round(value))
                return value
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return _coerce_numeric(matches[-1])


def _extract_int(text: str) -> Optional[int | float]:
    if not text:
        return None
    normalized = text.replace("−", "-")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]

    # For arithmetic prompts, many models put the answer as the first token/line.
    if lines:
        leading_match = re.match(
            r"^(?:####\s*)?(?:answer\s*[:=]\s*)?(?:final answer\s*[:=]\s*)?"
            r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*(?:\d{1,3}(?:,\d{3})+|\d+))?)",
            lines[0],
            flags=re.IGNORECASE,
        )
        if leading_match:
            return _extract_numeric_from_text(leading_match.group(1))

    for line in reversed(lines):
        if "####" in line:
            candidate = line.split("####", 1)[1]
            parsed = _extract_numeric_from_text(candidate)
            if parsed is not None:
                return parsed

    cue_patterns = [
        r"(?:final answer|answer|result)\s*(?:is|=|:)\s*([^\n]+)",
        r"=\s*([^\n]+)$",
    ]
    for pattern in cue_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            parsed = _extract_numeric_from_text(match.group(1))
            if parsed is not None:
                return parsed

    # Fallback: scan line-by-line from top so arithmetic first-line answers win over continuations.
    for line in lines:
        parsed = _extract_numeric_from_text(line)
        if parsed is not None:
            return parsed

    return None


def _stddev(values: List[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / len(values)) ** 0.5


def _generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 16) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def _generate(do_sample: bool, temperature: float) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if inputs["input_ids"].shape[-1] == 0:
            return ""
        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 1,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.eos_token_id,
                "use_cache": False,
                # Guard against bad logits from some model/hook combinations
                # so generation does not crash the entire experiment run.
                "remove_invalid_values": True,
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["renormalize_logits"] = True
            output_ids = model.generate(**generate_kwargs)
        new_token_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        if not text:
            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            text = decoded[len(prompt) :].strip()
        return text

    generated = _generate(do_sample=False, temperature=0.0)
    if not generated:
        # Deterministic decoding can immediately emit EOS for small models
        # (notably GPT-2).  Retry with mild sampling to force a token.
        try:
            generated = _generate(do_sample=True, temperature=0.7)
        except Exception:
            # Keep the evaluation running; an empty output is recorded as
            # unparseable instead of aborting the whole job.
            generated = ""
    return generated


def evaluate_bundle(model, tokenizer, bundle: DatasetBundle) -> Dict:
    results = []
    correct = 0
    evaluated = 0
    total = len(bundle.prompts)
    for idx, prompt in enumerate(bundle.prompts):
        answer = bundle.answers[idx] if bundle.answers else None
        generated = _generate_answer(model, tokenizer, prompt)
        prediction = _extract_int(generated)
        entry = {
            "prompt": prompt,
            "generated": generated,
            "parsed": prediction,
            "target": answer,
        }
        if answer is not None and prediction is not None:
            evaluated += 1
            if prediction == answer:
                correct += 1
            entry["correct"] = prediction == answer
        results.append(entry)
    accuracy = correct / evaluated if evaluated else None
    accuracy_all = (correct / total) if (bundle.answers is not None and total > 0) else None
    parse_rate = (evaluated / total) if total else 0.0
    return {
        "accuracy": accuracy,
        "accuracy_all": accuracy_all,
        "evaluated": evaluated,
        "total": total,
        "parse_rate": parse_rate,
        "results": results,
    }


@dataclass
class ExperimentPart:
    name: str
    type: str
    datasets: List[str]
    attention_layers: Optional[List[int]] = None
    attention_scales: Optional[List[float]] = None
    attention_head_targets: Optional[List[Dict[str, Any]]] = None
    attention_blends: Optional[List[Dict[str, Any]]] = None
    neuron_layers: Optional[List[int]] = None
    neuron_scales: Optional[List[float]] = None


class ExperimentRunner:
    def __init__(
        self,
        plan_path: Path,
        model_cache_dir: Path,
        results_dir: Path,
        override_devices: Optional[str] = None,
    ):
        self.plan = yaml.safe_load(Path(plan_path).read_text())
        self.model_cache_dir = Path(model_cache_dir)
        self.results_dir = Path(results_dir)
        self.override_devices = override_devices
        self.results_dir.mkdir(parents=True, exist_ok=True)
        dataset_options = self.plan.get("dataset_options", {})
        self.suite = load_tiered_suite(seed=self.plan.get("dataset_seed", 0), **dataset_options)

    def run(self) -> None:
        for model_cfg in self.plan["models"]:
            self._run_model(model_cfg)

    def _run_model(self, model_cfg: Dict) -> None:
        model_name = model_cfg["name"]
        devices = self.override_devices if self.override_devices is not None else model_cfg.get("devices", "")
        if devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
        model, tokenizer = load_local_model(
            model_name,
            cache_dir=model_cfg["cache_dir"],
            local_files_only=True,
            model_path=model_cfg.get("model_path"),
            tokenizer_path=model_cfg.get("tokenizer_path"),
        )
        model_dir = self.results_dir / model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        for part_cfg in self.plan["parts"]:
            part = ExperimentPart(**part_cfg)
            part_dir = model_dir / part.name
            part_dir.mkdir(parents=True, exist_ok=True)
            if part.type == "baseline":
                self._run_baseline_part(model, tokenizer, part, part_dir)
            elif part.type == "attention_sweep":
                self._run_attention_sweep(model, tokenizer, part, part_dir)
            elif part.type == "neuron_sweep":
                self._run_neuron_sweep(model, tokenizer, part, part_dir)
            else:
                raise ValueError(f"Unknown part type {part.type}")

    def _bundle(self, name: str) -> DatasetBundle:
        return self.suite.get(name)

    def _save_results(self, path: Path, payload: Dict) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _run_baseline_part(self, model, tokenizer, part: ExperimentPart, part_dir: Path) -> None:
        baseline_data = {}
        for dataset_name in part.datasets:
            bundle = self._bundle(dataset_name)
            run_results = []
            for run_idx in range(3):
                torch.manual_seed(run_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(run_idx)
                run_results.append(evaluate_bundle(model, tokenizer, bundle))

            baseline_data[dataset_name] = run_results[0]
            parsed_subset_scores = [
                float(result["accuracy"]) for result in run_results if result["accuracy"] is not None
            ]
            all_example_scores = [
                float(result["accuracy_all"]) for result in run_results if result["accuracy_all"] is not None
            ]
            parse_rates = [float(result["parse_rate"]) for result in run_results]
            baseline_data[dataset_name]["stability"] = {
                "metric": "accuracy",
                "scores": parsed_subset_scores,
                "mean": mean(parsed_subset_scores) if parsed_subset_scores else None,
                "std": _stddev(parsed_subset_scores) if parsed_subset_scores else None,
                "accuracy_all_scores": all_example_scores,
                "accuracy_all_mean": mean(all_example_scores) if all_example_scores else None,
                "accuracy_all_std": _stddev(all_example_scores) if all_example_scores else None,
                "parse_rates": parse_rates,
                "parse_rate_mean": mean(parse_rates) if parse_rates else None,
                "parse_rate_std": _stddev(parse_rates) if parse_rates else None,
            }
        self._save_results(part_dir / "baseline.json", baseline_data)

    def _run_attention_sweep(self, model, tokenizer, part: ExperimentPart, part_dir: Path) -> None:
        layers = part.attention_layers or []
        head_targets = part.attention_head_targets or []
        scale_entries: List[Any]
        blends_active = bool(part.attention_blends)
        if blends_active:
            scale_entries = part.attention_blends or []
        else:
            scale_entries = part.attention_scales or [1.0]
        sweep_records = []
        for entry in scale_entries:
            if blends_active:
                label = entry.get("label")
                module_scale = entry.get("module_scale", entry.get("scale", 1.0))
                head_scale_override = entry.get("head_scale")
                downscale_override = entry.get("downscale_others")
                record = {
                    "label": label or f"blend_{len(sweep_records)}",
                    "module_scale": module_scale,
                    "head_scale_override": head_scale_override,
                    "downscale_override": downscale_override,
                    "datasets": {},
                }
            else:
                module_scale = entry
                head_scale_override = None
                downscale_override = None
                record = {"scale": module_scale, "datasets": {}}

            configs = [AttentionHookConfig(layer=layer, head=None, scale=module_scale) for layer in layers]
            for target in head_targets:
                head_scale = target.get("scale")
                if head_scale is None:
                    head_scale = head_scale_override if head_scale_override is not None else module_scale
                downscale = target.get("downscale_others")
                if downscale_override is not None:
                    downscale = downscale_override
                cfg = AttentionHookConfig(
                    layer=target["layer"],
                    head=target.get("head"),
                    scale=head_scale,
                    downscale_others=downscale,
                )
                configs.append(cfg)
            with apply_hooks(model, attention_configs=configs):
                for dataset_name in part.datasets:
                    bundle = self._bundle(dataset_name)
                    eval_result = evaluate_bundle(model, tokenizer, bundle)
                    record["datasets"][dataset_name] = eval_result
            sweep_records.append(record)
        self._save_results(part_dir / "attention_sweep.json", {"records": sweep_records})

    def _run_neuron_sweep(self, model, tokenizer, part: ExperimentPart, part_dir: Path) -> None:
        layers = part.neuron_layers or []
        scales = part.neuron_scales or [1.0]
        sweep_records = []
        for scale in scales:
            configs = [NeuronHookConfig(layer=layer, neuron_index=0, scale=scale) for layer in layers]
            with apply_hooks(model, neuron_configs=configs):
                record = {"scale": scale, "datasets": {}}
                for dataset_name in part.datasets:
                    bundle = self._bundle(dataset_name)
                    eval_result = evaluate_bundle(model, tokenizer, bundle)
                    record["datasets"][dataset_name] = eval_result
                sweep_records.append(record)
        self._save_results(part_dir / "neuron_sweep.json", {"records": sweep_records})

    def _simple_metric(self, model, tokenizer, prompts: List[str], seed: int) -> float:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        scores = []
        for prompt in prompts[:5]:
            generated = _generate_answer(model, tokenizer, prompt)
            parsed = _extract_int(generated)
            scores.append(float(parsed or 0))
        return sum(scores) / len(scores) if scores else 0.0
