from __future__ import annotations

from report import report_markdown


def test_paper_runs_generate_paper_specific_report(tmp_path) -> None:
    rows = [
        {
            "model": "dense", "router_type": "none", "training_method": "supervised",
            "seed": 42, "target_density": 1.0, "lambda_density": 0.0,
            "val_loss": 2.0, "val_perplexity": 7.4, "val_accuracy": 0.4,
            "layers_per_token": 4.0, "compute_fraction": 1.0, "skip_fraction": 0.0,
            "estimated_executed_block_flops_per_sequence": 100.0,
            "paper_reproduction": True,
        },
        {
            "model": "sparse", "router_type": "linear", "training_method": "supervised",
            "seed": 42, "target_density": 0.5, "lambda_density": 0.1,
            "val_loss": 2.1, "val_perplexity": 8.2, "val_accuracy": 0.38,
            "layers_per_token": 4.0, "compute_fraction": 0.5, "skip_fraction": 0.5,
            "estimated_executed_block_flops_per_sequence": 106.0,
            "paper_reproduction": True,
        },
    ]
    report = report_markdown(rows, tmp_path)
    assert report.startswith("# SkipLayer Paper Reproduction")
    assert "0.1\\sum_l" in report
    assert "There is no density-loss warmup" in report
    assert "GRPO Objective" not in report


def test_mor_runs_generate_five_way_report(tmp_path) -> None:
    rows = [
        {
            "model": "dense", "router_type": "none", "training_method": "supervised",
            "seed": 42, "val_loss": 2.0, "val_perplexity": 7.4,
            "layers_per_token": 8.0, "skip_fraction": 0.0,
            "parameter_count": 2_000_000,
            "estimated_executed_block_flops_per_sequence": 200.0,
        },
        {
            "model": "mor", "router_type": "expert_linear",
            "training_method": "supervised", "seed": 42, "val_loss": 2.1,
            "val_perplexity": 8.2, "layers_per_token": 6.0, "skip_fraction": 0.25,
            "parameter_count": 1_000_000,
            "estimated_executed_block_flops_per_sequence": 140.0,
        },
    ]
    report = report_markdown(rows, tmp_path)
    assert report.startswith("# Mixture-of-Recursions")
    assert "Middle-Cycle" in report
    assert "oracle top-k" in report
    assert "MoR + GRPO" in report
