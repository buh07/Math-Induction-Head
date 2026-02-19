from src.statistics import summarize


def test_summarize_returns_effect_size():
    baseline = [0.5, 0.55, 0.6]
    intervention = [0.7, 0.72, 0.74]
    summary = summarize(baseline, intervention, num_bootstrap=100, seed=0)
    assert summary.mean > 0.7
    assert summary.effect_size > 0
    assert summary.ci_low <= summary.mean <= summary.ci_high
