from src.ablation import AblationStage, StagedAblationRunner


def test_staged_ablation_runner_applies_baselines():
    runner = StagedAblationRunner()
    activations = {
        17: [1.0, 2.0],
        18: [3.0, 5.0, 7.0],
    }
    stages = [
        AblationStage(name="stage0", layers=[17], baseline="zero"),
        AblationStage(name="stage1", layers=[17, 18], baseline="mean"),
    ]
    results = runner.run(activations, stages)
    assert results["stage0"]["17"] == [0.0, 0.0]
    assert results["stage1"]["17"] == [1.5, 1.5]
    assert results["stage1"]["18"] == [5.0, 5.0, 5.0]
