import torch 
import numpy as np
import os

from catboost import CatBoostClassifier, CatBoostRegressor

from neural_nets.mlp import MLP


# 一些配置项

# 离散动作空间相关配置
MODEL_PATH = 'trained_models/PPO_checkpoint_64803'
STATE_DIM = 2
ACTION_DIM = 1
HISTORY_LENGTH = 2
HIDDEN_SIZES = [32, 16]
TARGET = 0.768

# 离散动作权重
ACTION_WEIGHTS = [0.8, 0.95, 1, 1.05, 1.1, 1.2]
NUM_ACTIONS = len(ACTION_WEIGHTS)

FEATURE_WEIGHTS = [1, 1, 0.05 , 1]
TRAIN_VAL_RATIO = 0.8
REAL_DATA_PATHS = ['datas/filter2.csv']
FAIL_REAL_DATA_PATHS = ['datas/filter6.csv']
AUGMENT_BASE_K = 4
AUGMENT_K_MAX = 20
RARITY_ALPHA = 0.5
REAL_SAMPLE_WEIGHT = 8.0
FAIL_REAL_SAMPLE_WEIGHT = 2.0
REAL_AUG_SAMPLE_WEIGHT = 3.0
SYNTH_RATIO = 0.3
SYNTH_SAMPLE_WEIGHT = 1.0
OBS_BINS = [-1500, -1200, -1000, -800, -600, -400, -200, -100, -50, -10, -1, -0.1, -0.01, -0.004]
DISTILL_USE_PROBS = True


def get_train_data():
    """
    生成用于离散动作空间的训练数据
    """
    # 新采样逻辑：观测值区间自适应分布
    data_set_size = int(1e5)  # 采样量可调
    action_space = np.array(ACTION_WEIGHTS, dtype=np.float32)

    # 观测值分布区间（统一使用全局 OBS_BINS）
    obs_bins = np.array(OBS_BINS, dtype=np.float32)
    # 在每个区间内均匀采样
    obs_samples = []
    samples_per_bin = max(1, data_set_size // (len(obs_bins) - 1) // 2)
    for i in range(len(obs_bins) - 1):
        obs_samples.append(np.linspace(obs_bins[i], obs_bins[i + 1], samples_per_bin, endpoint=False))
    obs_samples = np.concatenate(obs_samples)
    # 观测值采样两组
    obs1 = np.random.choice(obs_samples, size=(data_set_size,))
    obs2 = np.random.choice(obs_samples, size=(data_set_size,))
    # 动作编号采样
    act1 = np.random.choice(action_space, size=(data_set_size,))
    act2 = np.random.choice(action_space, size=(data_set_size,))
    # 拼成 [动作编号1, 观测1, 动作编号2, 观测2]
    data = np.stack([act1, obs1, act2, obs2], axis=1)
    return data


def load_real_data(path):
    """
    Load real data from CSV or a log with 'states:tensor([...])' lines.
    Expected columns: [act1, obs1, act2, obs2]
    """
    try:
        data = np.loadtxt(path, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data.astype(np.float32)
    except Exception:
        pass

    import re
    rows = []
    number_re = re.compile(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', re.IGNORECASE)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'tensor' not in line and ',' not in line:
                continue
            nums = number_re.findall(line)
            if len(nums) >= 4:
                rows.append([float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])])
    if not rows:
        raise ValueError(f"未能从 {path} 解析出有效数据")
    return np.asarray(rows, dtype=np.float32)


def load_real_data_multi(paths):
    data_list = []
    for path in paths:
        if os.path.exists(path):
            data_list.append(load_real_data(path))
    if not data_list:
        return None
    return np.vstack(data_list)


def build_uniform_synth(real_data, ratio=0.3, seed=42):
    """
    Build uniform synthetic samples for obs columns based on fixed bins.
    Actions are sampled from ACTION_WEIGHTS.
    """
    rng = np.random.default_rng(seed)
    n_real = len(real_data)
    n_synth = max(1, int(n_real * ratio))
    bins = np.array(OBS_BINS, dtype=np.float32)
    bins = np.unique(bins)
    if len(bins) < 2:
        return np.empty((0, 4), dtype=np.float32)

    # sample obs uniformly per bin
    per_bin = int(np.ceil(n_synth / (len(bins) - 1)))
    obs = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        obs.append(rng.uniform(lo, hi, size=per_bin))
    obs = np.concatenate(obs)[:n_synth]

    act1 = rng.choice(ACTION_WEIGHTS, size=n_synth)
    act2 = rng.choice(ACTION_WEIGHTS, size=n_synth)
    obs1 = obs
    obs2 = obs
    X = np.stack([act1, obs1, act2, obs2], axis=1).astype(np.float32)
    return X

def compute_mag_bins(mags):
    """
    Compute magnitude bins for highly skewed obs distributions.
    """
    mags = np.asarray(mags, dtype=np.float32)
    qs = np.array([0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0], dtype=np.float32)
    edges = np.unique(np.quantile(mags, qs))
    if len(edges) < 4:
        edges = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 700.0, 1200.0], dtype=np.float32)
    edges[0] = 0.0
    return np.unique(edges)


def local_augment_dataset(X, base_k=4, k_max=20, rarity_alpha=0.5, seed=42):
    """
    Data-driven augmentation for skewed obs distributions. Preserves action
    values and adds noise to obs dimensions (1 and 3). Rare-magnitude bins
    receive more augmentation.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        return X

    obs_idx = (1, 3)
    mags = np.max(np.abs(X[:, obs_idx]), axis=1)
    edges = compute_mag_bins(mags)
    bin_ids = np.digitize(mags, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, len(edges) - 2)
    counts = np.bincount(bin_ids, minlength=len(edges) - 1)
    max_count = max(1, counts.max())

    bin_std = {}
    for idx in obs_idx:
        stds = np.zeros(len(edges) - 1, dtype=np.float32)
        for b in range(len(edges) - 1):
            mask = bin_ids == b
            if mask.any():
                stds[b] = float(np.std(X[mask, idx]))
        bin_std[idx] = stds

    aug = []
    for i in range(len(X)):
        x0 = X[i]
        b = int(bin_ids[i])
        rarity = (max_count / max(1, counts[b])) ** rarity_alpha
        k_i = int(np.clip(np.ceil(base_k * rarity), 1, k_max))
        for _ in range(k_i):
            x = x0.copy()
            for idx in obs_idx:
                mag = abs(x[idx])
                sigma = bin_std[idx][b]
                if mag < 0.01:
                    sigma = max(sigma, 1e-4)
                else:
                    sigma = max(sigma, 0.01 * mag, 0.2)
                x[idx] = x[idx] + rng.normal(0.0, sigma)
            aug.append(x)
    if not aug:
        return X
    return np.vstack([X, np.array(aug, dtype=np.float32)])

def get_eval_data():
    """
    Generate evaluation data similar to TABLE VI in the paper
    """
    under_utilized = np.array([1.1, -TARGET])
    steady_state = np.array([1.0, 0])
    congested = np.array([1.1, TARGET])


    under_utilized = {'name': 'under-utilized', 'data': under_utilized}
    on_target = {'name': 'on target', 'data': steady_state}
    congested = {'name': 'congested', 'data': congested}
    return [under_utilized, on_target, congested]


def nn_predict(nn_model, states):
    """
    神经网络推理，输出动作编号（离散）
    """
    with torch.no_grad():
        logits, _ = nn_model(torch.tensor(states).float())
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        action_indices = torch.argmax(logits, dim=-1)
    return action_indices.cpu().numpy()


def nn_predict_probs(nn_model, states):
    """
    神经网络推理，输出动作概率分布
    """
    with torch.no_grad():
        logits, _ = nn_model(torch.tensor(states).float())
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
    return probs.cpu().numpy()


if __name__ == '__main__':
    print("启动蒸馏流程")
    # 加载神经网络模型（只加载actor部分权重）
    print("加载神经网络模型")
    nn_model = MLP(input_size=STATE_DIM * HISTORY_LENGTH, output_size=NUM_ACTIONS, hidden_sizes=HIDDEN_SIZES,
                   activation='tanh', use_rnn=None, bias=False)

    # 加载PPO保存的完整checkpoint
    print("加载PPO checkpoint")
    checkpoint_state_dict = torch.load(MODEL_PATH)
    # 兼容直接保存actor/actor_critic的情况
    if 'model_state_dict' in checkpoint_state_dict:
        state_dict = checkpoint_state_dict['model_state_dict']
    else:
        state_dict = checkpoint_state_dict

    # 自动提取actor部分权重，并去除前缀（如'actor.net.'）
    actor_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('actor.net.net.'):
            new_k = k.replace('actor.net.net.', 'net.')
            actor_state_dict[new_k] = v
        elif k.startswith('actor.net.output_layer.'):
            new_k = k.replace('actor.net.output_layer.', 'output_layer.')
            actor_state_dict[new_k] = v
        elif k.startswith('actor.output_layer.linear.'):
            new_k = k.replace('actor.output_layer.linear.', 'output_layer.')
            actor_state_dict[new_k] = v
        elif k.startswith('actor.net.'):
            # 去掉'actor.net.'前缀
            new_k = k.replace('actor.net.', 'net.')
            actor_state_dict[new_k] = v
        elif k.startswith('actor.output_layer.'):
            # 去掉'actor.'前缀
            new_k = k.replace('actor.', '')
            actor_state_dict[new_k] = v
        # 兼容部分模型保存为'net.'或'output_layer.'
        elif k.startswith('net.') or k.startswith('output_layer.'):
            actor_state_dict[k] = v
    # 加载actor权重到MLP
    nn_model.load_state_dict(actor_state_dict, strict=False)
    nn_model.eval()

    # 生成训练数据
    print("准备训练数据")
    X_real_main = load_real_data_multi(REAL_DATA_PATHS)
    X_real_fail = load_real_data_multi(FAIL_REAL_DATA_PATHS)
    if X_real_main is not None or X_real_fail is not None:
        existing_paths = [p for p in REAL_DATA_PATHS + FAIL_REAL_DATA_PATHS if os.path.exists(p)]
        print(f"使用真实数据: {existing_paths}")
        base_real = X_real_main if X_real_main is not None else X_real_fail
        base_weight = REAL_SAMPLE_WEIGHT if X_real_main is not None else FAIL_REAL_SAMPLE_WEIGHT

        X_synth = build_uniform_synth(base_real, ratio=SYNTH_RATIO, seed=42)
        X_aug = local_augment_dataset(
            base_real,
            base_k=AUGMENT_BASE_K,
            k_max=AUGMENT_K_MAX,
            rarity_alpha=RARITY_ALPHA,
        )

        parts = [X_aug]
        weights_parts = []
        w_aug = np.ones(len(X_aug), dtype=np.float32)
        w_aug[:len(base_real)] = base_weight
        n_aug_only = len(X_aug) - len(base_real)
        if n_aug_only > 0:
            w_aug[len(base_real):len(base_real) + n_aug_only] = REAL_AUG_SAMPLE_WEIGHT
        weights_parts.append(w_aug)

        if X_real_fail is not None and X_real_main is not None:
            parts.append(X_real_fail)
            w_fail = np.full(len(X_real_fail), FAIL_REAL_SAMPLE_WEIGHT, dtype=np.float32)
            weights_parts.append(w_fail)

        if len(X_synth) > 0:
            parts.append(X_synth)
            w_synth = np.full(len(X_synth), SYNTH_SAMPLE_WEIGHT, dtype=np.float32)
            weights_parts.append(w_synth)

        X = np.vstack(parts)
        weights = np.concatenate(weights_parts)

        print(
            f"样本统计: real_main={0 if X_real_main is None else len(X_real_main)} | "
            f"real_fail={0 if X_real_fail is None else len(X_real_fail)} | "
            f"aug_total={len(X_aug)} | synth={len(X_synth)} | synth_ratio={SYNTH_RATIO}"
        )
        print("扩充训练数据已完成")
    else:
        print("真实数据不存在，使用合成数据")
        X = get_train_data()  # [动作编号1, 观测1, 动作编号2, 观测2]
        weights = None
    print("生成蒸馏监督标签")
    # 用“动作+观测”作为输入
    X_input = X  # shape: [N, 4]
    if DISTILL_USE_PROBS:
        y = nn_predict_probs(nn_model, X_input)  # [N, num_actions]
        unique_classes = np.unique(np.argmax(y, axis=-1))
    else:
        y = nn_predict(nn_model, X_input)  # 输出动作编号
        unique_classes = np.unique(y)
    print(f"监督标签类别数: {len(unique_classes)} | classes: {unique_classes.tolist()}")
    if len(unique_classes) < 2:
        print("警告：当前数据只激活到单一动作，补充合成数据以覆盖更多动作")
        X_extra = get_train_data()
        if DISTILL_USE_PROBS:
            y_extra = nn_predict_probs(nn_model, X_extra)
        else:
            y_extra = nn_predict(nn_model, X_extra)
        X_input = np.vstack([X_input, X_extra])
        y = np.concatenate([y, y_extra])
        if weights is not None:
            w_extra = np.ones(len(X_extra), dtype=np.float32)
            weights = np.concatenate([weights, w_extra])
        unique_classes = np.unique(np.argmax(y, axis=-1)) if DISTILL_USE_PROBS else np.unique(y)
        print(f"补充后类别数: {len(unique_classes)} | classes: {unique_classes.tolist()}")
        if len(unique_classes) < 2:
            raise RuntimeError("监督标签仍只有一个类别，无法训练分类器；请检查数据分布或NN策略是否退化")

    # 划分训练集和测试集（先打乱避免类别集中在同一段）
    print("打乱并划分训练/测试集")
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    X_input = X_input[perm]
    y = y[perm]
    if weights is not None:
        weights = weights[perm]

    split_idx = int(TRAIN_VAL_RATIO * len(y))
    X_tr, y_tr, X_test, y_test = X_input[:split_idx], y[:split_idx], X_input[split_idx:], y[split_idx:]
    if weights is not None:
        w_tr = weights[:split_idx]
    else:
        w_tr = None

    # 配置CatBoost（分类器/回归器）
    print("进入蒸馏阶段：训练CatBoost")
    if DISTILL_USE_PROBS:
        cb_params = {'loss_function': 'MultiRMSE',
                     'early_stopping_rounds': 50,
                     'verbose': 1,
                     'max_depth': 7,
                     'learning_rate': 1,
                     'iterations': 8,
                     'l2_leaf_reg': 1,
                     'min_data_in_leaf': 1,
                     'task_type': 'CPU',
                     'devices': "0:1",
                     'feature_weights': [1, 1, 1, 1]}
        cbr = CatBoostRegressor(**cb_params)
    else:
        cb_params = {'loss_function': 'MultiClass',
                     'early_stopping_rounds': 50,
                     'verbose': 1,
                     'max_depth': 6,
                     'learning_rate': 1,
                     'iterations': 8,
                     'l2_leaf_reg': 1,
                     'min_data_in_leaf': 1,
                     'task_type': 'CPU',
                     'devices': "0:1",
                     'feature_weights': [1, 1, 1, 1]}
        cbr = CatBoostClassifier(**cb_params)

    # 训练蒸馏模型
    cbr.fit(X_tr, y_tr, sample_weight=w_tr)
    print("蒸馏训练完成")

    print(f"Final model with {cbr.tree_count_} trees")
    if DISTILL_USE_PROBS:
        y_pred = np.array(cbr.predict(X_test))
        y_test = np.array(y_test)
        pred_idx = np.argmax(y_pred, axis=-1)
        true_idx = np.argmax(y_test, axis=-1)
        acc = (pred_idx == true_idx).mean()
        print(f"Top-1 Accuracy: {acc}")
    else:
        y_pred = np.array(cbr.predict(X_test)).flatten()
        y_test = np.array(y_test).flatten()
        acc = (np.round(y_pred).astype(int) == y_test).mean()
        print(f"Accuracy: {acc}")

    # 保存蒸馏后的CatBoost模型到文件，便于部署和查表化
    print("保存蒸馏模型")
    cbr.save_model('distilled_catboost_model.cbm')
    print("CatBoost模型已保存为 distilled_catboost_model.cbm")

    # 评估
    print("进入评估阶段")
    eval_states = get_eval_data()
    for prev_state in eval_states:
        for crnt_state in eval_states:
            print(f"current state: {crnt_state['name']} previous state: {prev_state['name']}")
            state = np.concatenate([crnt_state['data'], prev_state['data']], axis=-1)
            # 用“动作+观测”结构推理
            cb_pred = cbr.predict([state])
            nn_pred = nn_predict(nn_model, [state])
            if DISTILL_USE_PROBS:
                cb_idx = int(np.argmax(cb_pred[0]))
            else:
                cb_idx = int(np.round(cb_pred[0]))
            nn_idx = int(nn_pred[0])
            print(f"distilled action idx: {cb_idx}, value: {ACTION_WEIGHTS[cb_idx]} | neural network action idx: {nn_idx}, value: {ACTION_WEIGHTS[nn_idx]}")

    # # 推理延迟对比
    # import time
    # test_state = np.array([[1, 1.1, 0, 0.768]])  # 示例：动作编号1, 观测1.1, 动作编号0, 观测0.768
    # iterations = 10000

    # # --- 1. 评估神经网络 (NN) 耗时 ---
    # start_nn = time.time()
    # for _ in range(iterations):
    #     _ = nn_predict(nn_model, test_state)
    # end_nn = time.time()
    # avg_nn = (end_nn - start_nn) / iterations

    # # --- 2. 评估决策树 (CatBoost) 耗时 ---
    # start_cb = time.time()
    # for _ in range(iterations):
    #     _ = cbr.predict(test_state)
    # end_cb = time.time()
    # avg_cb = (end_cb - start_cb) / iterations

    # print(f"\n推理耗时对比 (平均每次):")
    # print(f"神经网络 (MLP): {avg_nn * 1e6:.2f} 微秒 (us)")
    # print(f"决策树 (CatBoost): {avg_cb * 1e6:.2f} 微秒 (us)")
    # print(f"加速比: {avg_nn / avg_cb:.1f} 倍")


#手动画出决策树的样子
#决策树贴回ns3模拟器中，测与baseline的fct对比
