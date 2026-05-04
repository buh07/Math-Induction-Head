"""Phase 2 operator-bucket datasets for arithmetic bottleneck experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .hash_utils import hash_strings


SUPPORTED_OPERATORS = ("addition", "subtraction", "multiplication")
SUPPORTED_BUCKETS: Dict[str, Tuple[str, ...]] = {
    "addition": ("no_carry", "single_carry", "cascading_carry"),
    "subtraction": ("no_borrow", "single_borrow", "cascading_borrow"),
    "multiplication": ("table_lookup", "partial_product", "carry_partial_sum"),
}

# Keep table-lookup buckets finite but large enough for high-count runs (e.g., 256 examples/bucket).
TABLE_LOOKUP_MAX_OPERAND = 19

_OPERATOR_SYMBOL = {
    "addition": "+",
    "subtraction": "-",
    "multiplication": "*",
}


@dataclass(frozen=True)
class OperatorBucketExample:
    prompt: str
    expected_answer: int | str
    operator: str
    bucket: str
    operands: List[int]
    representation_variant: str = "plain"
    metadata: Dict[str, Any] = field(default_factory=dict)
    answer_digits: Optional[List[int]] = None
    digit_targets: Optional[List[int]] = None
    carry_positions: Optional[List[int]] = None
    borrow_positions: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "operator": self.operator,
            "bucket": self.bucket,
            "operands": list(self.operands),
            "representation_variant": self.representation_variant,
            "metadata": dict(self.metadata),
            "answer_digits": list(self.answer_digits) if self.answer_digits is not None else None,
            "digit_targets": list(self.digit_targets) if self.digit_targets is not None else None,
            "carry_positions": list(self.carry_positions) if self.carry_positions is not None else None,
            "borrow_positions": list(self.borrow_positions) if self.borrow_positions is not None else None,
        }


@dataclass
class OperatorBucketDataset:
    operator: str
    bucket: str
    examples: List[OperatorBucketExample]
    seed: int
    representation_variant: str = "plain"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.operator}__{self.bucket}"

    @property
    def prompts(self) -> List[str]:
        return [example.prompt for example in self.examples]

    @property
    def targets(self) -> List[int | str]:
        return [example.expected_answer for example in self.examples]

    @property
    def dataset_hash(self) -> str:
        rows = []
        for example in self.examples:
            rows.append(json.dumps(example.to_dict(), sort_keys=True))
        return hash_strings(rows)

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "operator": self.operator,
            "bucket": self.bucket,
            "count": len(self.examples),
            "seed": self.seed,
            "representation_variant": self.representation_variant,
            "dataset_hash": self.dataset_hash,
            "metadata": dict(self.metadata),
        }


@dataclass
class OperatorBucketSuite:
    datasets: Dict[str, OperatorBucketDataset]
    suite_seed: int
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def names(self) -> List[str]:
        return sorted(self.datasets.keys())

    def all_bundles(self) -> List[OperatorBucketDataset]:
        return [self.datasets[name] for name in self.names()]

    def get(self, operator: str, bucket: Optional[str] = None) -> OperatorBucketDataset:
        key = operator if bucket is None else f"{operator}__{bucket}"
        if key not in self.datasets:
            raise KeyError(f"No operator bucket dataset named '{key}'")
        return self.datasets[key]

    def counts_by_operator(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for dataset in self.datasets.values():
            counts[dataset.operator] = counts.get(dataset.operator, 0) + len(dataset.examples)
        return counts

    def counts_by_bucket(self) -> Dict[str, int]:
        return {name: len(ds.examples) for name, ds in sorted(self.datasets.items())}

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": "operator_bucket_suite_v1",
            "suite_seed": self.suite_seed,
            "config_snapshot": self.config_snapshot,
            "datasets": [dataset.to_manifest_dict() for dataset in self.all_bundles()],
            "counts_by_operator": self.counts_by_operator(),
            "counts_by_bucket": self.counts_by_bucket(),
        }


def _digits_of_int(value: int) -> List[int]:
    if value == 0:
        return [0]
    return [int(ch) for ch in str(abs(value))]


def _format_prompt(a: int, b: int, symbol: str, *, representation_variant: str) -> str:
    if representation_variant == "plain":
        return f"Compute: {a} {symbol} {b} ="
    if representation_variant == "spaced_digits":
        def spaced(n: int) -> str:
            sign = "- " if n < 0 else ""
            digits = " ".join(list(str(abs(n))))
            return f"{sign}{digits}".strip()
        return f"Compute: {spaced(a)} {symbol} {spaced(b)} ="
    if representation_variant == "tagged_digits":
        def tagged(n: int) -> str:
            sign = "NEG " if n < 0 else ""
            digits = [f"d{i}:{ch}" for i, ch in enumerate(reversed(str(abs(n))))]
            return f"{sign}{' '.join(digits)}".strip()
        return f"Compute: {tagged(a)} {symbol} {tagged(b)} ="
    raise ValueError(f"Unsupported representation_variant: {representation_variant}")


def _add_answer_and_carries(a: int, b: int) -> Tuple[int, List[int]]:
    carry = 0
    carry_positions: List[int] = []
    place = 0
    aa = a
    bb = b
    while aa > 0 or bb > 0:
        da = aa % 10
        db = bb % 10
        total = da + db + carry
        carry_out = 1 if total >= 10 else 0
        if carry_out:
            carry_positions.append(place)
        carry = carry_out
        aa //= 10
        bb //= 10
        place += 1
    return a + b, carry_positions


def _classify_addition_bucket(a: int, b: int) -> str:
    _answer, carry_positions = _add_answer_and_carries(a, b)
    count = len(carry_positions)
    if count == 0:
        return "no_carry"
    if count == 1:
        return "single_carry"
    return "cascading_carry"


def _sub_answer_and_borrows(a: int, b: int) -> Tuple[int, List[int]]:
    if a < b:
        raise ValueError("Borrow analysis currently expects a >= b")
    borrow = 0
    borrow_positions: List[int] = []
    place = 0
    aa = a
    bb = b
    while aa > 0 or bb > 0:
        da = aa % 10
        db = bb % 10
        da -= borrow
        if da < db:
            da += 10
            borrow = 1
            borrow_positions.append(place)
        else:
            borrow = 0
        aa //= 10
        bb //= 10
        place += 1
    return a - b, borrow_positions


def _classify_subtraction_bucket(a: int, b: int) -> str:
    _answer, borrow_positions = _sub_answer_and_borrows(a, b)
    count = len(borrow_positions)
    if count == 0:
        return "no_borrow"
    if count == 1:
        return "single_borrow"
    return "cascading_borrow"


def _classify_multiplication_bucket(a: int, b: int) -> str:
    if a <= TABLE_LOOKUP_MAX_OPERAND and b <= TABLE_LOOKUP_MAX_OPERAND:
        return "table_lookup"
    if (a <= TABLE_LOOKUP_MAX_OPERAND) ^ (b <= TABLE_LOOKUP_MAX_OPERAND):
        return "partial_product"
    # crude but useful v1 heuristic: two multi-digit operands imply partial products + carry likely matter
    return "carry_partial_sum"


def _build_example(
    *,
    operator: str,
    bucket: str,
    a: int,
    b: int,
    representation_variant: str,
    seed: int,
    index: int,
) -> OperatorBucketExample:
    symbol = _OPERATOR_SYMBOL[operator]
    if operator == "addition":
        answer, carry_positions = _add_answer_and_carries(a, b)
        borrow_positions = None
    elif operator == "subtraction":
        answer, borrow_positions = _sub_answer_and_borrows(a, b)
        carry_positions = None
    elif operator == "multiplication":
        answer = a * b
        carry_positions = None
        borrow_positions = None
    else:
        raise ValueError(f"Unsupported operator: {operator}")

    answer_digits = _digits_of_int(answer)
    metadata = {
        "operator_symbol": symbol,
        "seed": seed,
        "index": index,
    }
    prompt = _format_prompt(a, b, symbol, representation_variant=representation_variant)
    return OperatorBucketExample(
        prompt=prompt,
        expected_answer=answer,
        operator=operator,
        bucket=bucket,
        operands=[a, b],
        representation_variant=representation_variant,
        metadata=metadata,
        answer_digits=answer_digits,
        digit_targets=list(answer_digits),
        carry_positions=carry_positions,
        borrow_positions=borrow_positions,
    )


def _candidate_operands(operator: str, rng: random.Random) -> Tuple[int, int]:
    if operator == "addition":
        return rng.randint(0, 999), rng.randint(0, 999)
    if operator == "subtraction":
        a = rng.randint(0, 999)
        b = rng.randint(0, 999)
        if b > a:
            a, b = b, a
        return a, b
    if operator == "multiplication":
        mode = rng.random()
        if mode < 0.33:
            return rng.randint(0, TABLE_LOOKUP_MAX_OPERAND), rng.randint(0, TABLE_LOOKUP_MAX_OPERAND)
        if mode < 0.66:
            if rng.random() < 0.5:
                return rng.randint(TABLE_LOOKUP_MAX_OPERAND + 1, 99), rng.randint(2, TABLE_LOOKUP_MAX_OPERAND)
            return rng.randint(2, TABLE_LOOKUP_MAX_OPERAND), rng.randint(TABLE_LOOKUP_MAX_OPERAND + 1, 99)
        return rng.randint(TABLE_LOOKUP_MAX_OPERAND + 1, 99), rng.randint(TABLE_LOOKUP_MAX_OPERAND + 1, 99)
    raise ValueError(f"Unsupported operator: {operator}")


def _classify_bucket(operator: str, a: int, b: int) -> str:
    if operator == "addition":
        return _classify_addition_bucket(a, b)
    if operator == "subtraction":
        return _classify_subtraction_bucket(a, b)
    if operator == "multiplication":
        return _classify_multiplication_bucket(a, b)
    raise ValueError(f"Unsupported operator: {operator}")


def generate_operator_bucket_dataset(
    operator: str,
    bucket: str,
    *,
    count: int,
    seed: int = 0,
    representation_variant: str = "plain",
    max_attempts_multiplier: int = 500,
) -> OperatorBucketDataset:
    if operator not in SUPPORTED_OPERATORS:
        raise ValueError(f"Unsupported operator '{operator}'")
    if bucket not in SUPPORTED_BUCKETS[operator]:
        raise ValueError(f"Unsupported bucket '{bucket}' for operator '{operator}'")
    if count <= 0:
        raise ValueError("count must be positive")

    rng = random.Random(seed)
    examples: List[OperatorBucketExample] = []
    seen_prompts: set[str] = set()
    attempts = 0
    max_attempts = max(count * max_attempts_multiplier, count + 100)
    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        a, b = _candidate_operands(operator, rng)
        if _classify_bucket(operator, a, b) != bucket:
            continue
        example = _build_example(
            operator=operator,
            bucket=bucket,
            a=a,
            b=b,
            representation_variant=representation_variant,
            seed=seed,
            index=len(examples),
        )
        if example.prompt in seen_prompts:
            continue
        seen_prompts.add(example.prompt)
        examples.append(example)

    if len(examples) < count:
        raise RuntimeError(
            f"Could not generate {count} unique examples for {operator}/{bucket}; got {len(examples)} after {attempts} attempts"
        )

    return OperatorBucketDataset(
        operator=operator,
        bucket=bucket,
        examples=examples,
        seed=seed,
        representation_variant=representation_variant,
        metadata={"attempts": attempts},
    )


def generate_operator_bucket_suite(
    operator_buckets: Mapping[str, Sequence[str]],
    *,
    counts_per_bucket: int,
    seed: int = 0,
    representation_variants: Optional[Sequence[str]] = None,
) -> OperatorBucketSuite:
    if counts_per_bucket <= 0:
        raise ValueError("counts_per_bucket must be positive")
    variants = list(representation_variants or ["plain"])
    if not variants:
        variants = ["plain"]

    datasets: Dict[str, OperatorBucketDataset] = {}
    suite_rng = random.Random(seed)
    for operator, buckets in operator_buckets.items():
        if operator not in SUPPORTED_OPERATORS:
            raise ValueError(f"Unsupported operator in suite config: {operator}")
        for bucket in buckets:
            if bucket not in SUPPORTED_BUCKETS[operator]:
                raise ValueError(f"Unsupported bucket '{bucket}' for operator '{operator}'")
            for variant_index, variant in enumerate(variants):
                ds_seed = suite_rng.randint(0, 2**31 - 1) + variant_index
                dataset = generate_operator_bucket_dataset(
                    operator,
                    bucket,
                    count=counts_per_bucket,
                    seed=ds_seed,
                    representation_variant=variant,
                )
                name = dataset.name if variant == "plain" else f"{dataset.name}__{variant}"
                datasets[name] = dataset
    return OperatorBucketSuite(
        datasets=datasets,
        suite_seed=seed,
        config_snapshot={
            "operator_buckets": {k: list(v) for k, v in operator_buckets.items()},
            "counts_per_bucket": counts_per_bucket,
            "representation_variants": variants,
        },
    )


def suite_diagnostics(suite: OperatorBucketSuite) -> Dict[str, Any]:
    per_dataset: Dict[str, Any] = {}
    for name, dataset in sorted(suite.datasets.items()):
        carry_examples = sum(1 for ex in dataset.examples if ex.carry_positions)
        borrow_examples = sum(1 for ex in dataset.examples if ex.borrow_positions)
        answer_lengths = [len(ex.answer_digits or []) for ex in dataset.examples]
        per_dataset[name] = {
            "operator": dataset.operator,
            "bucket": dataset.bucket,
            "count": len(dataset.examples),
            "representation_variant": dataset.representation_variant,
            "dataset_hash": dataset.dataset_hash,
            "carry_examples": carry_examples,
            "borrow_examples": borrow_examples,
            "answer_length_min": min(answer_lengths) if answer_lengths else 0,
            "answer_length_max": max(answer_lengths) if answer_lengths else 0,
            "sample_prompts": [ex.prompt for ex in dataset.examples[:3]],
        }
    return {
        "schema_version": "dataset_diagnostics_v1",
        "suite_seed": suite.suite_seed,
        "counts_by_operator": suite.counts_by_operator(),
        "counts_by_bucket": suite.counts_by_bucket(),
        "datasets": per_dataset,
    }


__all__ = [
    "SUPPORTED_OPERATORS",
    "SUPPORTED_BUCKETS",
    "OperatorBucketExample",
    "OperatorBucketDataset",
    "OperatorBucketSuite",
    "generate_operator_bucket_dataset",
    "generate_operator_bucket_suite",
    "suite_diagnostics",
]
