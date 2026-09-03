"""Generate and parse answers from a trained math checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from math_data import MathData, MathExample, extract_tagged_answer
from math_training_utils import generate_math_group
from model import SparseMoEMTPTransformer
from utils import load_checkpoint, select_device, set_seed


DEFAULT_QUESTIONS = [
    "Ava has 17 apples and buys 8 more. How many apples does she have?",
    "There are 6 boxes with 7 pencils in each. How many pencils are there?",
    "Patricia has 30 roses, gives away 24, then buys 15. How many roses does she have?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="experiments/math_v2_seed42/supervised/checkpoints/best_exact_match.pt",
    )
    parser.add_argument(
        "--question",
        dest="questions",
        action="append",
        help="Question to answer; repeat this option for multiple questions.",
    )
    parser.add_argument("--samples-per-question", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data-dir", default="data/gsm8k")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.samples_per_question < 1:
        parser.error("--samples-per-question must be positive")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists() and checkpoint_path.name == "best_exact_match.pt":
        fallback = checkpoint_path.with_name("best_val_loss.pt")
        if fallback.exists():
            checkpoint_path = fallback
    device = select_device(args.device)
    set_seed(args.seed)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if not isinstance(model, SparseMoEMTPTransformer):
        raise ValueError("checkpoint is not the math SkipLayer+MoE+MTP model")
    dataset_config = checkpoint.get("training_config", {}).get("dataset", {})
    if not dataset_config:
        dataset_config = checkpoint.get("training_config", {}).get("source_dataset", {})
    data = MathData(
        args.data_dir,
        model.config.context_length,
        seed=42,
        tokenizer_type=dataset_config.get("tokenizer_type", "byte"),
        bpe_vocab_size=int(dataset_config.get("bpe_vocab_size", 4096)),
    )
    if checkpoint["stoi"] != data.stoi:
        raise ValueError("checkpoint tokenizer does not match the math data tokenizer")

    questions = args.questions or DEFAULT_QUESTIONS
    for question_index, question in enumerate(questions, start=1):
        example = MathExample(question=question, reasoning="", answer="")
        set_seed(args.seed + question_index - 1)
        _, _, completions, depths = generate_math_group(
            model,
            data,
            example,
            args.samples_per_question,
            args.max_new_tokens,
            device,
            temperature=args.temperature,
        )
        print("=" * 80)
        print(f"Question: {question}")
        for sample_index, completion in enumerate(completions, start=1):
            parsed = extract_tagged_answer(completion)
            depth = float(depths[sample_index - 1].mean().item()) * model.config.n_layers
            print(f"\nSample {sample_index} | parsed answer: {parsed} | layers/token: {depth:.3f}")
            print(f"Response:{completion}")


if __name__ == "__main__":
    main()
