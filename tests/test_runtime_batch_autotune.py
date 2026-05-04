from src.runtime_batch_autotune import BatchAutotuneConfig, autotune_batch_size


def test_autotune_oom_backoff_selects_safe_batch():
    def probe(batch_size: int):
        if batch_size > 10:
            raise RuntimeError("CUDA out of memory")
        return {
            "metric_signature": [1.0, 2.0, 3.0],
            "logit_signature": [0.1, 0.2, 0.3],
        }

    result = autotune_batch_size(
        stage_name="localize",
        device="0",
        baseline_batch_size=8,
        run_probe_fn=probe,
        config=BatchAutotuneConfig(
            enabled=True,
            min_batch_size=4,
            max_batch_size=24,
            growth_factor=2.0,
            safety_backoff=0.9,
            max_retries_after_oom=1,
            equivalence_check_enabled=True,
        ),
    )
    assert result["enabled"] is True
    assert result["tuned_batch_size"] >= 4
    assert result["tuned_batch_size"] <= 8
    assert any(attempt["status"] == "oom" for attempt in result["attempts"])


def test_autotune_equivalence_guard_falls_back_to_baseline():
    def probe(batch_size: int):
        if batch_size > 16:
            raise RuntimeError("CUDA out of memory")
        return {
            "metric_signature": [float(batch_size)],
            "logit_signature": [float(batch_size)],
        }

    result = autotune_batch_size(
        stage_name="intervene",
        device="0",
        baseline_batch_size=8,
        run_probe_fn=probe,
        config=BatchAutotuneConfig(
            enabled=True,
            min_batch_size=4,
            max_batch_size=16,
            growth_factor=2.0,
            safety_backoff=1.0,
            equivalence_check_enabled=True,
            max_abs_logit_diff=1e-8,
            max_metric_diff=1e-8,
        ),
    )
    assert result["equivalence_check"]["passed"] is False
    assert result["status"] == "fallback_equivalence_failed"
    assert result["tuned_batch_size"] == 8
