"""High-level orchestration for multi-part experiments."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


NUMBER_PATTERN = re.compile(r"-?\d+")


def _extract_int(text: str) -> Optional[int]:
    match = NUMBER_PATTERN.search(text)
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None
    return None


def _generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 16) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def _generate(do_sample: bool, temperature: float) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if inputs["input_ids"].shape[-1] == 0:
            return ""
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
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
        generated = _generate(do_sample=True, temperature=0.7)
    return generated


def evaluate_bundle(model, tokenizer, bundle: DatasetBundle) -> Dict:
    results = []
    correct = 0
    evaluated = 0
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
    return {"accuracy": accuracy, "evaluated": evaluated, "results": results}


@dataclass
class ExperimentPart:
    name: str
    type: str
    datasets: List[str]
    attention_layers: Optional[List[int]] = None
    attention_scales: Optional[List[float]] = None
    neuron_layers: Optional[List[int]] = None
    neuron_scales: Optional[List[float]] = None


class ExperimentRunner:
    def __init__(
        self,
        plan_path: Path,
        model_cache_dir: Path,
        results_dir: Path,
    ):
        self.plan = yaml.safe_load(Path(plan_path).read_text())
        self.model_cache_dir = Path(model_cache_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.suite = load_tiered_suite(seed=self.plan.get("dataset_seed", 0))

    def run(self) -> None:
        for model_cfg in self.plan["models"]:
            self._run_model(model_cfg)

    def _run_model(self, model_cfg: Dict) -> None:
        model_name = model_cfg["name"]
        devices = model_cfg.get("devices", "")
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
        evaluator = BaselineEvaluator(
            evaluation_fn=lambda prompts, seed: self._simple_metric(model, tokenizer, prompts, seed)
        )
        for dataset_name in part.datasets:
            bundle = self._bundle(dataset_name)
            eval_result = evaluate_bundle(model, tokenizer, bundle)
            baseline_data[dataset_name] = eval_result
            report = evaluator.run(bundle, repeats=3)
            baseline_data[dataset_name]["stability"] = {
                "scores": report.scores,
                "mean": report.mean_score,
                "std": report.std_dev,
            }
        self._save_results(part_dir / "baseline.json", baseline_data)

    def _run_attention_sweep(self, model, tokenizer, part: ExperimentPart, part_dir: Path) -> None:
        layers = part.attention_layers or []
        scales = part.attention_scales or [1.0]
        sweep_records = []
        for scale in scales:
            configs = [AttentionHookConfig(layer=layer, head=None, scale=scale) for layer in layers]
            with apply_hooks(model, attention_configs=configs):
                record = {"scale": scale, "datasets": {}}
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
        rng = torch.Generator().manual_seed(seed)
        scores = []
        for prompt in prompts[:5]:
            generated = _generate_answer(model, tokenizer, prompt)
            parsed = _extract_int(generated)
            scores.append(float(parsed or 0))
        return sum(scores) / len(scores) if scores else 0.0
