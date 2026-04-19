import argparse
import numpy as np

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Visualize LUT slice for fixed act1/act2.")
    parser.add_argument("--lut", default="distilled_lut.npz", help="Path to LUT npz.")
    parser.add_argument("--act1-idx", type=int, default=2, help="Action index for act1.")
    parser.add_argument("--act2-idx", type=int, default=2, help="Action index for act2.")
    parser.add_argument("--out", default="lut_heatmap.png", help="Output image path.")
    args = parser.parse_args()

    data = np.load(args.lut)
    lut = data["lut"]
    obs_edges = data["obs_edges"]
    action_weights = data["action_weights"]

    if args.act1_idx < 0 or args.act1_idx >= lut.shape[0]:
        raise ValueError("act1_idx out of range.")
    if args.act2_idx < 0 or args.act2_idx >= lut.shape[2]:
        raise ValueError("act2_idx out of range.")

    slice_2d = lut[args.act1_idx, :, args.act2_idx, :]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(slice_2d, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_title(f"LUT heatmap (act1={action_weights[args.act1_idx]}, act2={action_weights[args.act2_idx]})")
    ax.set_xlabel("obs2 bin")
    ax.set_ylabel("obs1 bin")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("action index")

    ax.set_xticks([0, len(obs_edges) // 2, len(obs_edges) - 2])
    ax.set_yticks([0, len(obs_edges) // 2, len(obs_edges) - 2])
    ax.set_xticklabels(["low", "mid", "high"])
    ax.set_yticklabels(["low", "mid", "high"])

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved heatmap to {args.out}")


if __name__ == "__main__":
    main()
