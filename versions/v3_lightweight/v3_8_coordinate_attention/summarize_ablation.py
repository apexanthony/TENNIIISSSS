import argparse
import csv
import json
from pathlib import Path


ORDER = ["baseline", "ca", "hardneg", "aux", "full"]


def main():
    parser = argparse.ArgumentParser(description="Summarize five frozen V3.8 paper evaluations.")
    parser.add_argument("--root", default="exps/v38_essay_ablation")
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for variant in ORDER:
        run_dir = root / f"{variant}_seed{args.seed}"
        evaluation_path = run_dir / "paper_evaluation.json"
        if not evaluation_path.exists():
            print(f"pending: {evaluation_path}")
            continue
        report = json.loads(evaluation_path.read_text(encoding="utf-8"))
        test = report["test"]
        rows.append(
            {
                "variant": variant,
                "seed": args.seed,
                "selected_threshold": report["selected_threshold"],
                "accuracy": test["accuracy"],
                "precision": test["precision"],
                "recall": test["recall"],
                "f1": test["f1"],
                "tp": test["tp"],
                "wrong_localization": test["wrong_localization"],
                "missed_visible": test["missed_visible"],
                "fp_background": test["fp_background"],
                "tn": test["tn"],
            }
        )
    if not rows:
        raise RuntimeError("no completed paper_evaluation.json files found")
    output = root / "ablation_test_results.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output.resolve())
    for row in rows:
        print(
            f"{row['variant']:9s} Acc={row['accuracy']:.4f} P={row['precision']:.4f} "
            f"R={row['recall']:.4f} F1={row['f1']:.4f} thr={row['selected_threshold']:.2f}"
        )


if __name__ == "__main__":
    main()
