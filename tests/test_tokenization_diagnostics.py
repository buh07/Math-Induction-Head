from src.tokenization_diagnostics import analyze_prompts


def simple_tokenizer(text: str):
    return list(text)


def word_tokenizer(text: str):
    return text.split()


def test_analyze_prompts_handles_empty_input():
    report = analyze_prompts([], simple_tokenizer)
    assert report.total_numbers == 0
    assert report.average_tokens_per_prompt == 0.0


def test_analyze_prompts_counts_multi_token_numbers():
    prompts = ["12 + 3 =", "4 + 56 ="]
    report = analyze_prompts(prompts, word_tokenizer)
    assert report.total_numbers == 4
    assert report.single_token_numbers == 4  # word tokenizer keeps numbers intact
    report_chars = analyze_prompts(prompts, simple_tokenizer)
    assert report_chars.multi_token_numbers == 2  # only multi-digit numbers split
