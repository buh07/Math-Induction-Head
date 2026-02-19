from src.datasets import ArithmeticDataset, generate_prompt_batch


def test_arithmetic_dataset_generates_expected_count():
    dataset = ArithmeticDataset(num_problems=5, seed=1, operand_range=(0, 10))
    assert len(dataset.prompts()) == 5
    assert len(dataset.answers()) == 5


def test_arithmetic_dataset_is_reproducible():
    ds1 = ArithmeticDataset(num_problems=3, seed=123, operand_range=(0, 5))
    ds2 = ArithmeticDataset(num_problems=3, seed=123, operand_range=(0, 5))
    assert ds1.prompts() == ds2.prompts()
    assert ds1.answers() == ds2.answers()


def test_generate_prompt_batch_respects_operand_range():
    prompts = generate_prompt_batch(num_prompts=2, operand_range=(0, 1), seed=0)
    for prompt in prompts:
        parts = prompt.split()
        left, right = int(parts[0]), int(parts[2])
        assert left in {0, 1}
        assert right in {0, 1}
