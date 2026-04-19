import os
import numpy as np
import torch
import time

# Match PPO_AICC model structure for LUT generation.
from PPO_AICC.models.mlp import MLP
from PPO_AICC.models.model_utils import Categorical

# Configuration
MODEL_PATH = 'trained_models/PPO_checkpoint_64803'
STATE_DIM = 2
HISTORY_LENGTH = 2
HIDDEN_SIZES = [32, 16]

ACTION_WEIGHTS = [0.8, 0.95, 1, 1.05, 1.1, 1.2]
NUM_ACTIONS = len(ACTION_WEIGHTS)

# LUT binning config
OBS_RANGE = (-2000.0, 0.0)
NUM_BINS = 128

LUT_PATH = 'distilled_lut.npz'
STORE_PROBS = True
PROB_DTYPE = np.float16


def build_custom_bins():
    """
    Build piecewise-uniform bins with:
    - Wider bins near -1000
    - 2x finer bins in [-80, -20] and [-3, -2]
    """
    segments = [
        (-2000.0, -1200.0, 8),
        (-1200.0, -800.0, 6),   # widened near -1000
        (-800.0, -200.0, 24),
        (-200.0, -80.0, 16),
        (-80.0, -20.0, 32),     # 2x finer
        (-20.0, -3.0, 20),
        (-3.0, -2.0, 8),        # 2x finer
        (-2.0, 0.0, 14),
    ]
    edges = []
    for lo, hi, n in segments:
        if n <= 0:
            raise ValueError("Segment bin count must be positive.")
        seg = np.linspace(lo, hi, n + 1, endpoint=True)
        if edges:
            seg = seg[1:]
        edges.append(seg)
    edges = np.concatenate(edges).astype(np.float32)
    return edges


def bin_centers(edges):
    return (edges[:-1] + edges[1:]) * 0.5


class ActorLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(
            input_size=STATE_DIM * HISTORY_LENGTH,
            output_size=HIDDEN_SIZES[-1],
            hidden_sizes=HIDDEN_SIZES,
            activation='tanh',
            use_rnn=None,
            bias=False,
        )
        self.output_layer = Categorical(HIDDEN_SIZES[-1], NUM_ACTIONS)

    def forward(self, x: torch.Tensor):
        x, _ = self.net(x)
        return self.output_layer(x)


def load_actor_model():
    nn_model = ActorLike()

    checkpoint_state_dict = torch.load(MODEL_PATH, map_location='cpu')
    if 'model_state_dict' in checkpoint_state_dict:
        state_dict = checkpoint_state_dict['model_state_dict']
    else:
        state_dict = checkpoint_state_dict

    actor_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('actor.net.net.'):
            new_k = k.replace('actor.net.net.', 'net.')
            actor_state_dict[new_k] = v
        elif k.startswith('actor.net.output_layer.'):
            new_k = k.replace('actor.net.output_layer.', 'output_layer.')
            actor_state_dict[new_k] = v
        elif k.startswith('actor.net.'):
            new_k = k.replace('actor.net.', 'net.')
            actor_state_dict[new_k] = v
        elif k.startswith('actor.output_layer.'):
            new_k = k.replace('actor.', '')
            actor_state_dict[new_k] = v
        elif k.startswith('net.') or k.startswith('output_layer.'):
            actor_state_dict[k] = v

    nn_model.load_state_dict(actor_state_dict, strict=False)
    nn_model.eval()
    return nn_model


def nn_predict(nn_model, states):
    with torch.no_grad():
        dist = nn_model(torch.tensor(states).float())
        action_indices = dist.probs.argmax(dim=-1)
    return action_indices.cpu().numpy()


def nn_predict_probs(nn_model, states):
    with torch.no_grad():
        dist = nn_model(torch.tensor(states).float())
        probs = dist.probs
    return probs.cpu().numpy()


def build_lut(nn_model, obs_edges, action_weights, store_probs=False):
    if len(obs_edges) != NUM_BINS + 1:
        raise ValueError("obs_edges length must be NUM_BINS + 1.")
    obs_centers = bin_centers(obs_edges)
    num_bins = len(obs_centers)
    num_actions = len(action_weights)
    if num_actions != NUM_ACTIONS:
        raise ValueError("ACTION_WEIGHTS length must match NUM_ACTIONS.")
    if num_actions > np.iinfo(np.uint8).max:
        raise ValueError("NUM_ACTIONS too large for uint8 LUT storage.")

    lut = np.zeros((num_actions, num_bins, num_actions, num_bins), dtype=np.uint8)
    lut_probs = None
    if store_probs:
        lut_probs = np.zeros(
            (num_actions, num_bins, num_actions, num_bins, num_actions),
            dtype=PROB_DTYPE,
        )

    # Full enumeration over all bins using bin centers as representative values.
    for a1_idx, act1 in enumerate(action_weights):
        for b1_idx, obs1 in enumerate(obs_centers):
            for a2_idx, act2 in enumerate(action_weights):
                obs2 = obs_centers
                act1_col = np.full_like(obs2, act1, dtype=np.float32)
                act2_col = np.full_like(obs2, act2, dtype=np.float32)
                obs1_col = np.full_like(obs2, obs1, dtype=np.float32)
                states = np.stack([act1_col, obs1_col, act2_col, obs2], axis=1)
                if store_probs:
                    probs = nn_predict_probs(nn_model, states).astype(PROB_DTYPE)
                    lut_probs[a1_idx, b1_idx, a2_idx, :, :] = probs
                    preds = np.argmax(probs, axis=-1).astype(np.uint8)
                else:
                    preds = nn_predict(nn_model, states).astype(np.uint8)
                lut[a1_idx, b1_idx, a2_idx, :] = preds

    return lut, lut_probs


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
    if STATE_DIM * HISTORY_LENGTH != 4:
        raise ValueError("Expected STATE_DIM * HISTORY_LENGTH == 4 for [act1, obs1, act2, obs2].")

    print("Loading actor model")
    t0 = time.time()
    nn_model = load_actor_model()
    t1 = time.time()

    print("Building obs bins")
    obs_edges = build_custom_bins()
    if len(obs_edges) != NUM_BINS + 1:
        raise ValueError(f"Expected {NUM_BINS + 1} edges, got {len(obs_edges)}")

    print("Generating LUT")
    t2 = time.time()
    lut, lut_probs = build_lut(nn_model, obs_edges, ACTION_WEIGHTS, store_probs=STORE_PROBS)
    t3 = time.time()

    print(f"Saving LUT to {LUT_PATH}")
    save_kwargs = dict(
        lut=lut,
        obs_edges=obs_edges,
        action_weights=np.array(ACTION_WEIGHTS, dtype=np.float32),
    )
    if lut_probs is not None:
        save_kwargs["lut_probs"] = lut_probs
    np.savez(LUT_PATH, **save_kwargs)
    t4 = time.time()

    print(f"Timing: load_model={t1 - t0:.3f}s | build_lut={t3 - t2:.3f}s | save={t4 - t3:.3f}s")
    print("Done")
