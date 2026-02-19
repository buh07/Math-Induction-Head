"""Dataset utilities for generating synthetic arithmetic prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .hash_utils import hash_strings

Operation = Tuple[str, callable]


def _default_operations() -> Sequence[Operation]:
    return (
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    )


@dataclass
class ArithmeticExample:
    prompt: str
    answer: int


@dataclass
class ArithmeticDataset:
    """Generate reproducible arithmetic prompts."""

    num_problems: int = 10
    operand_range: Tuple[int, int] = (0, 100)
    operations: Sequence[Operation] = field(default_factory=_default_operations)
    seed: int = 0
    include_answer: bool = False

    def __post_init__(self) -> None:
        low, high = self.operand_range
        if low >= high:
            raise ValueError("operand_range lower bound must be < upper bound")
        self._rng = random.Random(self.seed)
        self.examples: List[ArithmeticExample] = []
        for _ in range(self.num_problems):
            a = self._rng.randint(low, high)
            b = self._rng.randint(low, high)
            op_symbol, op_fn = self._rng.choice(self.operations)
            answer = op_fn(a, b)
            prompt = f"{a} {op_symbol} {b} ="
            if self.include_answer:
                prompt = f"{prompt} {answer}"
            self.examples.append(ArithmeticExample(prompt=prompt, answer=answer))

    def prompts(self) -> List[str]:
        return [example.prompt for example in self.examples]

    def answers(self) -> List[int]:
        return [example.answer for example in self.examples]


def generate_prompt_batch(
    num_prompts: int,
    operand_range: Tuple[int, int] = (0, 100),
    seed: int = 0,
) -> List[str]:
    dataset = ArithmeticDataset(
        num_problems=num_prompts, operand_range=operand_range, seed=seed
    )
    return dataset.prompts()


@dataclass
class MultiOperationArithmeticDataset:
    num_problems: int = 50
    operand_range: Tuple[int, int] = (0, 500)
    max_operations: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.prompts: List[str] = []
        self.answers: List[int] = []
        ops = _default_operations()
        for _ in range(self.num_problems):
            expression = []
            values = []
            result = self._rng.randint(*self.operand_range)
            expression.append(str(result))
            values.append(result)
            current_value = result
            for _ in range(self._rng.randint(1, self.max_operations)):
                op_symbol, op_fn = self._rng.choice(ops)
                operand = self._rng.randint(*self.operand_range)
                expression.append(op_symbol)
                expression.append(str(operand))
                current_value = op_fn(current_value, operand)
            prompt = "Compute: " + " ".join(expression) + " ="
            self.prompts.append(prompt)
            self.answers.append(current_value)


@dataclass
class GSMStyleDataset:
    num_problems: int = 40
    seed: int = 0

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        templates = [
            "{name} has {a} apples and buys {b} more. After giving {c} to a friend, how many remain?",
            "A school printed {a} worksheets per class for {b} classes. If {c} were damaged, how many usable worksheets are left?",
            "A train travels {a} km in the morning and {b} km in the afternoon. If it needs to reach {c} km total, how many km remain?",
        ]
        names = ["Alice", "Ben", "Cara", "Diego", "Eva"]
        self.prompts: List[str] = []
        self.answers: List[int] = []
        for _ in range(self.num_problems):
            template = rng.choice(templates)
            a = rng.randint(20, 120)
            b = rng.randint(10, 80)
            c = rng.randint(5, 60)
            name = rng.choice(names)
            prompt = template.format(name=name, a=a, b=b, c=c)
            if "apples" in template:
                answer = a + b - c
            elif "worksheets" in template:
                answer = a * b - c
            else:
                answer = max(c - (a + b), 0)
            self.prompts.append(prompt)
            self.answers.append(answer)


@dataclass
class DatasetSpec:
    name: str
    operand_range: Tuple[int, int]
    num_problems: int
    operations: Sequence[Operation] = field(default_factory=_default_operations)
    seed: int = 0


@dataclass
class DatasetBundle:
    name: str
    prompts: List[str]
    answers: Optional[List[int]] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def dataset_hash(self) -> str:
        return hash_strings(self.prompts)


@dataclass
class TieredDatasetSuite:
    bundles: Dict[str, DatasetBundle]

    def names(self) -> List[str]:
        return list(self.bundles.keys())

    def hashes(self) -> Dict[str, str]:
        return {name: bundle.dataset_hash for name, bundle in self.bundles.items()}

    def get(self, name: str) -> DatasetBundle:
        if name not in self.bundles:
            raise KeyError(f"No dataset named '{name}' in suite")
        return self.bundles[name]


def _generate_symbolic_prompts(num_prompts: int, rng: random.Random) -> List[str]:
    templates = [
        lambda a, b, c: f"Pattern: {a}{b}{a}{b} -> ?",
        lambda a, b, c: f"Mirror: {a}{b}{c}{b}{a} -> ?",
        lambda a, b, c: f"Bracket completion: <{a}|{b}{c}> -> ?",
        lambda a, b, c: f"Sequence: {a}, {b}, {c}, ... what comes next?",
    ]
    alphabet = list("XYZUVWABCDEF")
    prompts: List[str] = []
    for _ in range(num_prompts):
        fn = rng.choice(templates)
        args = [rng.choice(alphabet) for _ in range(3)]
        prompts.append(fn(*args))
    return prompts


def load_tiered_suite(seed: int = 0) -> TieredDatasetSuite:
    specs = [
        DatasetSpec(
            name="tier1_in_distribution",
            operand_range=(0, 100),
            num_problems=100,
            seed=seed,
        ),
        DatasetSpec(
            name="tier2_near_ood",
            operand_range=(1000, 2000),
            num_problems=60,
            seed=seed + 1,
        ),
    ]
    bundles: Dict[str, DatasetBundle] = {}
    for spec in specs:
        dataset = ArithmeticDataset(
            num_problems=spec.num_problems,
            operand_range=spec.operand_range,
            operations=spec.operations,
            seed=spec.seed,
        )
        bundles[spec.name] = DatasetBundle(
            name=spec.name,
            prompts=dataset.prompts(),
            answers=dataset.answers(),
            metadata={
                "type": "arithmetic",
                "operand_range": f"{spec.operand_range}",
                "num_problems": str(spec.num_problems),
            },
        )

    symbolic_rng = random.Random(seed + 2)
    symbolic_prompts = _generate_symbolic_prompts(40, symbolic_rng)
    bundles["tier_symbolic_patterns"] = DatasetBundle(
        name="tier_symbolic_patterns",
        prompts=symbolic_prompts,
        metadata={
            "type": "symbolic",
            "num_problems": "40",
        },
    )

    multi_dataset = MultiOperationArithmeticDataset(seed=seed + 3)
    bundles["tier_multi_operation"] = DatasetBundle(
        name="tier_multi_operation",
        prompts=multi_dataset.prompts,
        answers=multi_dataset.answers,
        metadata={
            "type": "multi_operation",
            "num_problems": str(multi_dataset.num_problems),
            "max_operations": str(multi_dataset.max_operations),
        },
    )

    gsm_dataset = GSMStyleDataset(seed=seed + 4)
    bundles["tier_gsm_style"] = DatasetBundle(
        name="tier_gsm_style",
        prompts=gsm_dataset.prompts,
        answers=gsm_dataset.answers,
        metadata={
            "type": "gsm_style",
            "num_problems": str(gsm_dataset.num_problems),
        },
    )
    return TieredDatasetSuite(bundles=bundles)
