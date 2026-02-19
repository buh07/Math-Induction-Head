from src.datasets import DatasetBundle
from src.evaluation import BaselineEvaluator, run_parameter_sweep


def test_baseline_evaluator_reports_mean_and_std():
    bundle = DatasetBundle(name="dummy", prompts=["1 + 2 ="])

    def eval_fn(prompts, seed):
        return float(len(prompts[0]) + seed)

    evaluator = BaselineEvaluator(eval_fn)
    report = evaluator.run(bundle, repeats=2, seed_offset=0)
    assert report.scores == [float(len(bundle.prompts[0])), float(len(bundle.prompts[0]) + 1)]
    assert report.mean_score > 0
    assert report.std_dev > 0


def test_run_parameter_sweep_covers_grid():
    grid = {"a": [0.0, 1.0], "b": [2.0]}

    def sweep_fn(params):
        return params["a"] + params["b"]

    results = run_parameter_sweep(grid, sweep_fn)
    assert len(results) == 2
    assert results[0].params.keys() == {"a", "b"}
