from src.datasets import (
    DatasetBundle,
    TieredDatasetSuite,
    load_tiered_suite,
    MultiOperationArithmeticDataset,
    GSMStyleDataset,
    _parse_gsm8k_answer,
)


def test_load_tiered_suite_produces_expected_tiers():
    suite = load_tiered_suite(seed=0)
    names = set(suite.names())
    assert "tier1_in_distribution" in names
    assert "tier2_near_ood" in names
    assert "tier_symbolic_patterns" in names
    assert "tier_multi_operation" in names
    assert "tier_gsm_style" in names
    hashes = suite.hashes()
    assert hashes["tier1_in_distribution"] != hashes["tier2_near_ood"]


def test_dataset_bundle_hash_changes_with_prompts():
    bundle_a = DatasetBundle(name="a", prompts=["1 + 2 ="])
    bundle_b = DatasetBundle(name="b", prompts=["1 + 3 ="])
    assert bundle_a.dataset_hash != bundle_b.dataset_hash


def test_multi_operation_dataset_generates_multi_step_prompts():
    dataset = MultiOperationArithmeticDataset(num_problems=5, seed=0)
    assert len(dataset.prompts) == 5
    assert all(prompt.startswith("Compute:") for prompt in dataset.prompts)


def test_gsm_style_dataset_generates_word_problems():
    dataset = GSMStyleDataset(num_problems=3, seed=0)
    assert len(dataset.prompts) == 3
    assert all("?" in prompt for prompt in dataset.prompts)


def test_parse_gsm8k_answer_handles_currency_and_fraction():
    value = _parse_gsm8k_answer("#### $1,234 dollars")
    assert value == 1234
    half = _parse_gsm8k_answer("#### 3/2 hours")
    assert abs(half - 1.5) < 1e-9
