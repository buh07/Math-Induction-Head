from scripts.phase2.run_operator_bottleneck_suite import (
    _build_intervention_anomaly_report,
    _evaluate_cot_gate,
    _derive_localization_thresholds,
    _derive_specificity_threshold_from_random_controls,
    _evaluate_intervention_sanity_gate,
    _evaluate_localization_gate,
    _evaluate_specificity_gate,
)


def test_localization_gate_fails_on_zero_effects():
    localization_index = {
        "addition::attention_heads": {
            "status": "ok",
            "metrics": [
                {
                    "effect_nonzero_rate": 0.0,
                    "answer_token_prob_delta_mean": 1e-8,
                }
            ],
            "robustness_summary": {"same_set_shuffle_invariance": 0.9},
        }
    }
    gate = _evaluate_localization_gate(
        localization_index,
        nonzero_min=0.01,
        prob_delta_abs_min=1e-5,
        required_robustness_modes=["same_set_shuffle_invariance"],
        require_all_runs=True,
    )
    assert gate["passes"] is False
    assert gate["checks"][0]["passes"] is False


def test_localization_gate_fails_when_required_robustness_missing():
    localization_index = {
        "addition::mlp_neurons": {
            "status": "ok",
            "metrics": [
                {
                    "effect_nonzero_rate": 0.2,
                    "answer_token_prob_delta_mean": 2e-4,
                }
            ],
            "robustness_summary": {"same_set_shuffle_invariance": None},
        }
    }
    gate = _evaluate_localization_gate(
        localization_index,
        nonzero_min=0.01,
        prob_delta_abs_min=1e-5,
        required_robustness_modes=["same_set_shuffle_invariance"],
        require_all_runs=True,
    )
    assert gate["passes"] is False
    assert gate["checks"][0]["missing_robustness_modes"] == ["same_set_shuffle_invariance"]


def test_localization_gate_fails_when_answer_target_coverage_too_low():
    localization_index = {
        "addition::attention_heads": {
            "status": "ok",
            "metrics": [
                {
                    "effect_nonzero_rate": 0.2,
                    "answer_token_prob_delta_mean": 5e-4,
                    "metadata": {"answer_valid_count": 100},
                }
            ],
            "prompt_set": {"answer_target_valid_rate": 0.15},
            "robustness_summary": {"same_set_shuffle_invariance": 0.9},
        }
    }
    gate = _evaluate_localization_gate(
        localization_index,
        nonzero_min=0.01,
        prob_delta_abs_min=1e-5,
        required_robustness_modes=["same_set_shuffle_invariance"],
        require_all_runs=True,
        min_answer_target_valid_rate=0.5,
    )
    assert gate["passes"] is False
    assert "low_target_coverage" in gate["checks"][0]["failure_reasons"]


def test_specificity_gate_requires_non_target_evidence_when_enabled():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "results": [
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "ablation", "scale": 0.0},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.2, "ci": [-0.28, -0.12]}},
                        }
                    },
                }
            ],
        }
    }
    gate = _evaluate_specificity_gate(
        payloads,
        ci_low_min=0.01,
        mean_gap_min=0.01,
        require_non_target_operator_evidence=True,
    )
    assert gate["passes"] is False
    assert gate["checks"][0]["best_non_target"] is None
    assert gate["checks"][0]["failure_reason"] in {
        "missing_non_target_evidence_for_intervention",
        "insufficient_non_target_scope",
    }


def test_specificity_gate_passes_with_target_and_non_target_gap():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "results": [
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "ablation", "scale": 0.0},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.25, "ci": [-0.35, -0.12]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.02, "ci": [-0.07, 0.03]}},
                        },
                    },
                }
            ],
        }
    }
    gate = _evaluate_specificity_gate(
        payloads,
        ci_low_min=0.01,
        mean_gap_min=0.05,
        require_non_target_operator_evidence=True,
    )
    assert gate["passes"] is True


def test_specificity_gate_fails_when_both_primary_interventions_required_but_missing():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "results": [
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "ablation", "scale": 0.0},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.2, "ci": [-0.28, -0.12]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.01, "ci": [-0.03, 0.01]}},
                        },
                    },
                }
            ],
        }
    }
    gate = _evaluate_specificity_gate(
        payloads,
        ci_low_min=0.01,
        mean_gap_min=0.01,
        require_non_target_operator_evidence=True,
        require_primary_set=True,
        primary_set_name="top",
        require_both_primary_interventions=True,
        primary_scales=[0.0, 1.25],
        primary_k_values=[5],
    )
    assert gate["passes"] is False
    assert gate["checks"][0]["failure_reason"] == "missing_target_evidence_for_intervention"


def test_specificity_gate_rejects_positive_ablation_delta_under_signed_policy():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "results": [
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "ablation", "scale": 0.0},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": 0.18, "ci": [0.11, 0.25]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": 0.02, "ci": [-0.02, 0.06]}},
                        },
                    },
                }
            ],
        }
    }
    gate = _evaluate_specificity_gate(
        payloads,
        ci_low_min=0.01,
        mean_gap_min=0.01,
        require_non_target_operator_evidence=True,
        require_directionality=True,
    )
    assert gate["passes"] is False
    assert gate["checks"][0]["failure_reason"] == "target_ci_low_below_threshold"


def test_specificity_gate_rejects_negative_amplification_delta_under_signed_policy():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "results": [
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "amplification", "scale": 1.25},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.12, "ci": [-0.20, -0.05]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.01, "ci": [-0.06, 0.03]}},
                        },
                    },
                }
            ],
        }
    }
    gate = _evaluate_specificity_gate(
        payloads,
        ci_low_min=0.01,
        mean_gap_min=0.01,
        require_non_target_operator_evidence=True,
    )
    assert gate["passes"] is False
    assert gate["checks"][0]["failure_reason"] == "target_ci_low_below_threshold"


def test_specificity_gate_does_not_cherry_pick_best_condition_only():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "results": [
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "ablation", "scale": 0.0},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.35, "ci": [-0.48, -0.20]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.08, "ci": [-0.15, -0.01]}},
                        },
                    },
                },
                {
                    "condition": {"component_set_name": "K10:top", "intervention": "ablation", "scale": 0.0},
                    "datasets": {
                        "addition__single_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.06, "ci": [-0.10, 0.02]}},
                        },
                        "subtraction__single_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.02, "ci": [-0.06, 0.02]}},
                        },
                    },
                },
            ],
        }
    }
    gate = _evaluate_specificity_gate(
        payloads,
        ci_low_min=0.01,
        mean_gap_min=0.01,
        require_non_target_operator_evidence=True,
    )
    assert gate["passes"] is False
    # One preregistered row has weak/zero-crossing support, so aggregate support fails.
    assert gate["checks"][0]["failure_reason"] == "target_ci_low_below_threshold"


def test_intervention_sanity_gate_blocks_flagged_payloads():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "sanity_summary": {
                "total_flagged_datasets": 2,
                "flag_type_counts": {"impossible_jump_near_floor_to_high_accuracy": 2},
            },
        }
    }
    gate = _evaluate_intervention_sanity_gate(payloads, enabled=True)
    assert gate["passes"] is False
    assert gate["checks"][0]["flagged_datasets"] == 2


def test_specificity_gate_uses_primary_only_q_values_for_multiplicity_blocking():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "results": [
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "ablation", "scale": 0.0},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.20, "ci": [-0.30, -0.10]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.03, "ci": [-0.08, 0.02]}},
                        },
                    },
                },
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "amplification", "scale": 1.25},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": 0.08, "ci": [0.02, 0.13]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": 0.01, "ci": [-0.04, 0.05]}},
                        },
                    },
                },
            ],
            "analysis": {
                "multiplicity_report": {
                    "rows": [
                        {
                            "condition": "K5:top",
                            "operator": "addition",
                            "dataset": "addition__no_carry",
                            "set_label": "top",
                            "intervention": "ablation",
                            "scale": 0.0,
                            "is_primary_comparison": True,
                            "q_value": 0.8,
                            "q_value_primary": 0.02,
                        },
                        {
                            "condition": "K5:top",
                            "operator": "addition",
                            "dataset": "addition__no_carry",
                            "set_label": "top",
                            "intervention": "amplification",
                            "scale": 1.25,
                            "is_primary_comparison": True,
                            "q_value": 0.7,
                            "q_value_primary": 0.03,
                        },
                    ]
                }
            },
        }
    }
    gate = _evaluate_specificity_gate(
        payloads,
        ci_low_min=0.01,
        mean_gap_min=0.01,
        require_non_target_operator_evidence=True,
        require_primary_set=True,
        primary_set_name="top",
        require_both_primary_interventions=True,
        multiplicity_blocking_enabled=True,
        multiplicity_q_max=0.05,
    )
    assert gate["passes"] is True
    assert gate["checks"][0]["multiplicity_summary"]["best_primary_q"] == 0.02


def test_specificity_gate_multiplicity_requires_all_primary_q_values_below_threshold():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "results": [
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "ablation", "scale": 0.0},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.20, "ci": [-0.30, -0.10]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.03, "ci": [-0.08, 0.02]}},
                        },
                    },
                },
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "amplification", "scale": 1.25},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": 0.08, "ci": [0.02, 0.13]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": 0.01, "ci": [-0.04, 0.05]}},
                        },
                    },
                },
            ],
            "analysis": {
                "multiplicity_report": {
                    "rows": [
                        {
                            "condition": "K5:top",
                            "operator": "addition",
                            "dataset": "addition__no_carry",
                            "set_label": "top",
                            "intervention": "ablation",
                            "scale": 0.0,
                            "is_primary_comparison": True,
                            "q_value_primary": 0.02,
                        },
                        {
                            "condition": "K5:top",
                            "operator": "addition",
                            "dataset": "addition__no_carry",
                            "set_label": "top",
                            "intervention": "amplification",
                            "scale": 1.25,
                            "is_primary_comparison": True,
                            "q_value_primary": 0.20,
                        },
                    ]
                }
            },
        }
    }
    gate = _evaluate_specificity_gate(
        payloads,
        ci_low_min=0.01,
        mean_gap_min=0.01,
        require_non_target_operator_evidence=True,
        require_primary_set=True,
        primary_set_name="top",
        require_both_primary_interventions=True,
        multiplicity_blocking_enabled=True,
        multiplicity_q_max=0.05,
    )
    assert gate["passes"] is False
    mult = gate["checks"][0]["multiplicity_summary"]
    assert mult["best_primary_q"] == 0.02
    assert mult["worst_primary_q"] == 0.20
    assert mult["failure_reason"] == "primary_q_above_threshold"


def test_specificity_gate_multiplicity_requires_complete_primary_q_coverage():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "results": [
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "ablation", "scale": 0.0},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.25, "ci": [-0.35, -0.15]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.03, "ci": [-0.08, 0.02]}},
                        },
                    },
                },
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "amplification", "scale": 1.25},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": 0.10, "ci": [0.03, 0.18]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": 0.01, "ci": [-0.04, 0.05]}},
                        },
                    },
                },
            ],
            "analysis": {
                "multiplicity_report": {
                    "rows": [
                        {
                            "condition": "K5:top",
                            "operator": "addition",
                            "dataset": "addition__no_carry",
                            "set_label": "top",
                            "intervention": "ablation",
                            "scale": 0.0,
                            "q_value_primary": 0.02,
                            "is_primary_comparison": True,
                        }
                        # amplification row intentionally missing -> coverage failure
                    ]
                }
            },
        }
    }
    gate = _evaluate_specificity_gate(
        payloads,
        ci_low_min=0.01,
        mean_gap_min=0.01,
        require_non_target_operator_evidence=True,
        require_primary_set=True,
        primary_set_name="top",
        require_both_primary_interventions=True,
        primary_interventions=["ablation", "amplification"],
        primary_scales=[0.0, 1.25],
        primary_k_values=[5],
        multiplicity_blocking_enabled=True,
        multiplicity_q_max=0.05,
        multiplicity_require_complete_primary_coverage=True,
    )
    assert gate["passes"] is False
    mult = gate["checks"][0]["multiplicity_summary"]
    assert mult["failure_reason"] == "missing_primary_q_coverage"
    assert mult["n_expected_primary_q_rows"] == 2
    assert mult["n_observed_primary_q_rows"] == 1


def test_specificity_gate_multiplicity_filters_to_primary_scales():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "results": [
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "ablation", "scale": 0.0},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.2, "ci": [-0.3, -0.1]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.01, "ci": [-0.04, 0.02]}},
                        },
                    },
                },
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "amplification", "scale": 1.25},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": 0.08, "ci": [0.02, 0.14]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": 0.01, "ci": [-0.03, 0.05]}},
                        },
                    },
                },
            ],
            "analysis": {
                "multiplicity_report": {
                    "rows": [
                        {
                            "condition": "K5:top",
                            "operator": "addition",
                            "dataset": "addition__no_carry",
                            "set_label": "top",
                            "intervention": "ablation",
                            "scale": 0.0,
                            "q_value_primary": 0.01,
                            "is_primary_comparison": True,
                        },
                        {
                            "condition": "K5:top",
                            "operator": "addition",
                            "dataset": "addition__no_carry",
                            "set_label": "top",
                            "intervention": "amplification",
                            "scale": 1.25,
                            "q_value_primary": 0.02,
                            "is_primary_comparison": True,
                        },
                        {
                            "condition": "K5:top",
                            "operator": "addition",
                            "dataset": "addition__no_carry",
                            "set_label": "top",
                            "intervention": "amplification",
                            "scale": 1.5,
                            "q_value_primary": 0.9,
                            "is_primary_comparison": True,
                        },
                    ]
                }
            },
        }
    }
    gate = _evaluate_specificity_gate(
        payloads,
        ci_low_min=0.01,
        mean_gap_min=0.01,
        require_non_target_operator_evidence=True,
        require_primary_set=True,
        primary_set_name="top",
        require_both_primary_interventions=True,
        primary_interventions=["ablation", "amplification"],
        primary_scales=[0.0, 1.25],
        primary_k_values=[5],
        multiplicity_blocking_enabled=True,
        multiplicity_q_max=0.05,
        multiplicity_require_complete_primary_coverage=True,
    )
    assert gate["passes"] is True
    mult = gate["checks"][0]["multiplicity_summary"]
    assert mult["worst_primary_q"] == 0.02


def test_specificity_gate_primary_k_filter_excludes_non_preregistered_k_rows():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "results": [
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "ablation", "scale": 0.0},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.2, "ci": [-0.3, -0.1]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.02, "ci": [-0.06, 0.02]}},
                        },
                    },
                },
                {
                    "condition": {"component_set_name": "K10:top", "intervention": "ablation", "scale": 0.0},
                    "datasets": {
                        "addition__single_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.05, "ci": [-0.1, 0.02]}},
                        },
                        "subtraction__single_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": -0.02, "ci": [-0.06, 0.02]}},
                        },
                    },
                },
            ],
        }
    }
    gate = _evaluate_specificity_gate(
        payloads,
        ci_low_min=0.01,
        mean_gap_min=0.01,
        require_non_target_operator_evidence=True,
        primary_interventions=["ablation"],
        primary_scales=[0.0],
        primary_k_values=[5],
        require_both_primary_interventions=False,
    )
    assert gate["passes"] is True
    row_checks = gate["checks"][0]["row_checks"]
    assert len(row_checks) == 1
    assert row_checks[0]["row_key"]["k_value"] == 5


def test_specificity_gate_absolute_sign_policy_ci_crossing_zero_fails_support():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "results": [
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "amplification", "scale": 1.25},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": 0.10, "ci": [-0.02, 0.22]}},
                        },
                        "subtraction__no_borrow": {
                            "operator": "subtraction",
                            "metrics": {"delta_vs_random_accuracy_all": {"mean": 0.01, "ci": [-0.04, 0.05]}},
                        },
                    },
                }
            ],
        }
    }
    gate = _evaluate_specificity_gate(
        payloads,
        ci_low_min=0.01,
        mean_gap_min=0.01,
        require_non_target_operator_evidence=True,
        sign_policy="absolute",
    )
    assert gate["passes"] is False
    assert gate["checks"][0]["failure_reason"] == "target_ci_low_below_threshold"


def test_intervention_anomaly_report_collects_flagged_samples():
    payloads = {
        "addition::attention_heads": {
            "status": "ok",
            "results": [
                {
                    "condition": {"component_set_name": "K5:top", "intervention": "ablation", "scale": 0.0},
                    "sanity_flags": ["ablation_large_positive_delta_requires_leakage_review"],
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "bucket": "no_carry",
                            "sanity_flags": ["ablation_large_positive_delta_requires_leakage_review"],
                            "metrics": {
                                "accuracy_all": 0.95,
                                "parse_rate": 1.0,
                                "delta_vs_baseline_accuracy_all": {"mean": 0.8, "ci": [0.7, 0.9]},
                                "delta_vs_random_accuracy_all": {"mean": 0.75, "ci": [0.6, 0.85]},
                            },
                            "prediction_samples": [
                                {"generated": "42", "parsed": 42, "target": 42},
                                {"generated": "13", "parsed": 13, "target": 14},
                            ],
                        }
                    },
                }
            ],
        }
    }
    report = _build_intervention_anomaly_report(payloads, sample_cap_per_dataset=1)
    assert report["schema_version"] == "phase2_intervention_anomaly_report_v1"
    assert report["summary"]["total_flagged_conditions"] == 1
    assert report["summary"]["total_flagged_datasets"] == 1
    entry = report["runs"]["addition::attention_heads"]["flagged_conditions"][0]["datasets"]["addition__no_carry"]
    assert len(entry["prediction_samples"]) == 1
    assert entry["flags"] == ["ablation_large_positive_delta_requires_leakage_review"]


def test_calibrated_thresholds_respect_configured_floors():
    loc_thresholds = _derive_localization_thresholds(
        {
            "runA": {
                "metrics": [
                    {"answer_token_prob_delta_mean": 1e-8, "effect_nonzero_rate": 0.0},
                    {"answer_token_prob_delta_mean": 2e-8, "effect_nonzero_rate": 0.001},
                ]
            }
        },
        policy="target_shuffle",
        quantile=0.95,
        nonzero_floor=0.02,
        abs_prob_floor=1e-5,
    )
    assert loc_thresholds["localization_nonzero_min"] >= 0.02
    assert loc_thresholds["localization_prob_delta_abs_min"] >= 1e-5

    spec_thresholds = _derive_specificity_threshold_from_random_controls(
        {
            "runA": {
                "results": [
                    {
                        "condition": {"component_set_name": "K5:random_matched"},
                        "datasets": {
                            "addition__no_carry": {
                                "metrics": {"delta_vs_baseline_accuracy_all": {"mean": 0.003}},
                            }
                        },
                    }
                ]
            }
        },
        quantile=0.95,
        ci_low_floor=0.01,
    )
    assert spec_thresholds["specificity_ci_low_min"] >= 0.01


def test_cot_gate_fails_when_parse_rate_shift_explains_effect():
    payload = {
        "addition": {
            "status": "ok",
            "direct_metrics": {"parse_rate": 0.5},
            "cot_metrics": {"parse_rate": 1.0},
            "sensitivity_deltas": {
                "baseline_direct_vs_cot": {
                    "accuracy_all_delta": 0.2,
                    "parse_rate_delta": 0.5,
                }
            },
        }
    }
    gate = _evaluate_cot_gate(payload, effect_abs_min=0.01, parse_rate_delta_abs_max=0.05)
    assert gate["passes"] is False
    assert gate["checks"][0]["failure_reason"] == "parse_rate_delta_too_large"


def test_cot_gate_passes_on_substantive_effect_with_parse_control():
    payload = {
        "addition": {
            "status": "ok",
            "direct_metrics": {"parse_rate": 0.95},
            "cot_metrics": {"parse_rate": 0.96},
            "sensitivity_deltas": {
                "baseline_direct_vs_cot": {
                    "accuracy_all_delta": 0.08,
                    "parse_rate_delta": 0.01,
                }
            },
        }
    }
    gate = _evaluate_cot_gate(payload, effect_abs_min=0.01, parse_rate_delta_abs_max=0.05)
    assert gate["passes"] is True


def test_cot_gate_fails_when_strict_ci_policy_enabled_without_excluding_zero():
    payload = {
        "addition": {
            "status": "ok",
            "n_pairs": 40,
            "direct_metrics": {"parse_rate": 0.95},
            "cot_metrics": {"parse_rate": 0.96},
            "sensitivity_deltas": {
                "baseline_direct_vs_cot": {
                    "accuracy_all_delta": 0.08,
                    "accuracy_all_delta_ci": [-0.01, 0.12],
                    "parse_rate_delta": 0.01,
                }
            },
        }
    }
    gate = _evaluate_cot_gate(
        payload,
        effect_abs_min=0.01,
        parse_rate_delta_abs_max=0.05,
        min_pairs=32,
        parse_rate_min=0.8,
        require_accuracy_ci_excludes_zero=True,
    )
    assert gate["passes"] is False
    assert gate["checks"][0]["failure_reason"] == "accuracy_ci_includes_zero"
