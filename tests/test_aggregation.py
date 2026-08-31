from __future__ import annotations

from aggregate_results import grouping_value


def test_missing_group_keys_share_stable_value() -> None:
    assert grouping_value({}, "lambda_grpo") is None
    assert grouping_value({"lambda_grpo": ""}, "lambda_grpo") is None
    assert grouping_value({"lambda_grpo": float("nan")}, "lambda_grpo") is None
    assert grouping_value({"lambda_grpo": 0.1}, "lambda_grpo") == 0.1
