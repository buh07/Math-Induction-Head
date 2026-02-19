"""Tokenization diagnostics for arithmetic prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

TokenizerFn = Callable[[str], Sequence[str]]

NUMBER_PATTERN = re.compile(r"\d+")


@dataclass
class TokenizationReport:
    total_numbers: int
    single_token_numbers: int
    multi_token_numbers: int
    average_tokens_per_prompt: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_numbers": self.total_numbers,
            "single_token_numbers": self.single_token_numbers,
            "multi_token_numbers": self.multi_token_numbers,
            "average_tokens_per_prompt": self.average_tokens_per_prompt,
        }


def analyze_prompts(
    prompts: Iterable[str],
    tokenize: TokenizerFn,
) -> TokenizationReport:
    prompts = list(prompts)
    if not prompts:
        return TokenizationReport(0, 0, 0, 0.0)

    token_counts = [len(tokenize(prompt)) for prompt in prompts]
    avg_tokens = sum(token_counts) / len(token_counts)

    total_numbers = 0
    single_token_numbers = 0
    multi_token_numbers = 0

    for prompt in prompts:
        numbers = NUMBER_PATTERN.findall(prompt)
        for number in numbers:
            total_numbers += 1
            tokenized_number = tokenize(number)
            if len(tokenized_number) <= 1:
                single_token_numbers += 1
            else:
                multi_token_numbers += 1

    return TokenizationReport(
        total_numbers=total_numbers,
        single_token_numbers=single_token_numbers,
        multi_token_numbers=multi_token_numbers,
        average_tokens_per_prompt=avg_tokens,
    )
