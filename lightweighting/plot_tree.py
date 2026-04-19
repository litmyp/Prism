import argparse
import os
import shutil
import subprocess

from catboost import CatBoostClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Visualize CatBoost tree via Graphviz.")
    p.add_argument(
        "--model",
        default="distilled_catboost_model.cbm",
        help="Path to CatBoost .cbm model",
    )
    p.add_argument(
        "--tree-idx",
        type=int,
        default=None,
        help="Tree index to export (default: export all trees)",
    )
    p.add_argument(
        "--format",
        choices=["dot", "png", "svg"],
        default="svg",
        help="Output format (dot/png/svg)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output file path (optional)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model_path = args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = CatBoostClassifier()
    model.load_model(model_path)
    tree_count = model.tree_count_

    def export_one(tidx, out_path):
        dot_path = out_path if args.format == "dot" else f"tree_{tidx}.dot"
        graph = model.plot_tree(tree_idx=tidx)
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write(graph.source)
        if args.format == "dot":
            print(f"Wrote {dot_path}")
            return
        dot_bin = shutil.which("dot")
        if not dot_bin:
            print("Graphviz 'dot' not found. Install graphviz or use --format dot.")
            print(f"Wrote {dot_path}")
            return
        cmd = [dot_bin, f"-T{args.format}", dot_path, "-o", out_path]
        subprocess.check_call(cmd)
        print(f"Wrote {out_path}")

    if args.tree_idx is not None:
        out_path = args.out
        if out_path is None:
            out_path = f"tree_{args.tree_idx}.{args.format}"
        export_one(args.tree_idx, out_path)
        return

    for tidx in range(tree_count):
        out_path = f"tree_{tidx}.{args.format}"
        export_one(tidx, out_path)


if __name__ == "__main__":
    main()
