import torch 
import numpy as np
import os

from catboost import CatBoostClassifier, CatBoostRegressor

from distill_network import get_train_data, build_uniform_synth, local_augment_dataset, OBS_BINS

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
REAL_DATA_PATHS = ['datas/filter_mix_states.csv']#['datas/filter_mix_states.csv']
FAIL_REAL_DATA_PATHS = []
LABELED_DATA_PATHS = []
AUGMENT_BASE_K = 4
AUGMENT_K_MAX = 20
RARITY_ALPHA = 0.5
REAL_SAMPLE_WEIGHT = 8.0
FAIL_REAL_SAMPLE_WEIGHT = 2.0
REAL_AUG_SAMPLE_WEIGHT = 3.0
SYNTH_RATIO = 0.3
SYNTH_SAMPLE_WEIGHT = 1.0
DISTILL_USE_PROBS = False
EVAL_RANDOM_SAMPLES = 12
EVAL_RANDOM_SEED = 42
USE_CLASS_WEIGHTS = True
USE_AUGMENT = False
USE_BOUNDARY_AUGMENT = True
USE_DRIFT_CORRECTION = True
BOUNDARY_THRESHOLD = -200.0
BOUNDARY_N_SAMPLES = 2000
BOUNDARY_NOISE_STD = 15.0
DRIFT_RANGE = 0.05
DRIFT_N_SAMPLES = 2000
AUG_SAMPLE_WEIGHT = 0.5
BALANCE_MIN_COUNT = 0
BALANCE_MAX_ATTEMPTS = 500000
BALANCE_OBS_NOISE = 0.02
ARCHIVE_PATH = 'datas/filter_archieve.csv'


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


def load_labeled_data_multi(paths):
    data_list = []
    for path in paths:
        if os.path.exists(path):
            data_list.append(load_real_data(path))
    if not data_list:
        return None, None
    data = np.vstack(data_list)
    if data.shape[1] < 5:
        raise ValueError("带标签数据需要 5 列: [act1, obs1, act2, obs2, action]")
    X = data[:, :4].astype(np.float32)
    y_raw = data[:, 4].astype(np.float32)
    # 若标签是动作值，映射到最接近的动作索引
    weights = np.array(ACTION_WEIGHTS, dtype=np.float32)
    if not np.all(np.isin(y_raw, np.arange(len(weights), dtype=np.float32))):
        y = np.array([int(np.argmin(np.abs(weights - v))) for v in y_raw], dtype=np.int64)
    else:
        y = y_raw.astype(np.int64)
    return X, y


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
        states_arr = np.asarray(states, dtype=np.float32)
        logits, _ = nn_model(torch.tensor(states_arr).float())
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        action_indices = torch.argmax(logits, dim=-1)
    return action_indices.cpu().numpy()


def nn_predict_probs(nn_model, states):
    """
    神经网络推理，输出动作概率分布
    """
    with torch.no_grad():
        states_arr = np.asarray(states, dtype=np.float32)
        logits, _ = nn_model(torch.tensor(states_arr).float())
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
    return probs.cpu().numpy()


def boundary_augment(real_data, obs_indices=(1, 3), threshold=-200.0, n_samples=2000, noise_std=15.0, seed=42):
    """
    在危险区(obs > threshold)附近制造更多样本。
    """
    rng = np.random.default_rng(seed)
    real_data = np.asarray(real_data, dtype=np.float32)
    danger_mask = np.any(real_data[:, obs_indices] > threshold, axis=1)
    danger_pool = real_data[danger_mask]
    if len(danger_pool) == 0:
        return np.empty((0, real_data.shape[1]), dtype=np.float32)
    indices = rng.choice(len(danger_pool), size=n_samples, replace=True)
    aug_X = danger_pool[indices].copy()
    aug_X[:, obs_indices] += rng.normal(0.0, noise_std, size=(n_samples, len(obs_indices)))
    return aug_X


def drift_correction(real_data, act_indices=(0, 2), drift_range=0.05, n_samples=2000, seed=42):
    """
    模拟历史动作的漂移，学习纠偏。
    """
    rng = np.random.default_rng(seed)
    real_data = np.asarray(real_data, dtype=np.float32)
    indices = rng.choice(len(real_data), size=n_samples, replace=True)
    aug_X = real_data[indices].copy()
    multipliers = rng.uniform(1 - drift_range, 1 + drift_range, size=(n_samples, len(act_indices)))
    aug_X[:, act_indices] *= multipliers
    # 保持动作值在合法范围内
    aug_X[:, act_indices] = np.clip(aug_X[:, act_indices], min(ACTION_WEIGHTS), max(ACTION_WEIGHTS))
    return aug_X

def balance_minor_actions(nn_model, X_input, y, min_count=500, max_attempts=500000, obs_noise=0.02, seed=42):
    """
    Balance classes by sampling from existing states with small noise on obs,
    re-sampling actions if needed until NN outputs the target action.
    """
    rng = np.random.default_rng(seed)
    X_input = np.asarray(X_input, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    counts = np.bincount(y, minlength=NUM_ACTIONS)
    target_actions = [i for i, c in enumerate(counts) if c < min_count]
    if not target_actions:
        return X_input, y

    X_aug = []
    y_aug = []
    for target in target_actions:
        attempts = 0
        while counts[target] < min_count and attempts < max_attempts:
            attempts += 1
            # sample a base state
            base = X_input[rng.integers(0, len(X_input))].copy()
            # resample actions (indices 0 and 2) from action weights
            base[0] = float(rng.choice(ACTION_WEIGHTS))
            base[2] = float(rng.choice(ACTION_WEIGHTS))
            # add small noise to obs (indices 1 and 3)
            base[1] = base[1] + rng.normal(0.0, obs_noise * max(1.0, abs(base[1])))
            base[3] = base[3] + rng.normal(0.0, obs_noise * max(1.0, abs(base[3])))
            pred = int(nn_predict(nn_model, [base])[0])
            if pred == target:
                X_aug.append(base)
                y_aug.append(pred)
                counts[target] += 1
        if counts[target] < min_count:
            print(f"警告: 动作 {target} 未能补到 {min_count}（当前 {counts[target]}），已达最大尝试次数")

    if X_aug:
        X_input = np.vstack([X_input, np.asarray(X_aug, dtype=np.float32)])
        y = np.concatenate([y, np.asarray(y_aug, dtype=np.int64)])
    return X_input, y


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
    X_labeled, y_labeled = load_labeled_data_multi(LABELED_DATA_PATHS)
    if X_labeled is not None and y_labeled is not None:
        existing_paths = [p for p in LABELED_DATA_PATHS if os.path.exists(p)]
        print(f"使用带标签真实数据: {existing_paths}")
        X_input = X_labeled
        y = y_labeled
        weights = None
    else:
        X_real_main = load_real_data_multi(REAL_DATA_PATHS)
        X_real_fail = load_real_data_multi(FAIL_REAL_DATA_PATHS)
        if X_real_main is not None or X_real_fail is not None:
            existing_paths = [p for p in REAL_DATA_PATHS + FAIL_REAL_DATA_PATHS if os.path.exists(p)]
            print(f"使用真实数据: {existing_paths}")
            base_real = X_real_main if X_real_main is not None else X_real_fail
            base_weight = REAL_SAMPLE_WEIGHT if X_real_main is not None else FAIL_REAL_SAMPLE_WEIGHT

            if USE_AUGMENT:
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
                X = base_real
                weights = None
                print(
                    f"样本统计: real_main={0 if X_real_main is None else len(X_real_main)} | "
                    f"real_fail={0 if X_real_fail is None else len(X_real_fail)} | "
                    "aug_total=0 | synth=0 | synth_ratio=0"
                )
                print("未进行扩充/合成")
        else:
            print("真实数据不存在，使用合成数据")
            X = get_train_data()  # [动作编号1, 观测1, 动作编号2, 观测2]
            weights = None

        print("生成蒸馏监督标签")
        # 用“动作+观测”作为输入
        X_input = X  # shape: [N, 4]
        y = nn_predict(nn_model, X_input)  # 输出动作编号

    # 可选增强：边界增强 + 动作纠偏（重标注）
    if USE_BOUNDARY_AUGMENT or USE_DRIFT_CORRECTION:
        aug_parts = []
        if USE_BOUNDARY_AUGMENT:
            aug_parts.append(
                boundary_augment(
                    X_input,
                    threshold=BOUNDARY_THRESHOLD,
                    n_samples=BOUNDARY_N_SAMPLES,
                    noise_std=BOUNDARY_NOISE_STD,
                )
            )
        if USE_DRIFT_CORRECTION:
            aug_parts.append(
                drift_correction(
                    X_input,
                    drift_range=DRIFT_RANGE,
                    n_samples=DRIFT_N_SAMPLES,
                )
            )
        aug_X = np.vstack([p for p in aug_parts if len(p) > 0]) if aug_parts else np.empty((0, X_input.shape[1]))
        if len(aug_X) > 0:
            y_aug = nn_predict(nn_model, aug_X)
            X_input = np.vstack([X_input, aug_X])
            y = np.concatenate([y, y_aug])
            if weights is None:
                weights = np.ones(len(y), dtype=np.float32)
            weights[-len(y_aug):] = AUG_SAMPLE_WEIGHT

    unique_classes = np.unique(y)
    print(f"监督标签类别数: {len(unique_classes)} | classes: {unique_classes.tolist()}")
    counts = np.bincount(y, minlength=NUM_ACTIONS)
    print(f"action counts: {counts.tolist()}")
    missing = [i for i, c in enumerate(counts) if c == 0]
    if missing:
        print(f"missing idx: {missing} | values: {[ACTION_WEIGHTS[i] for i in missing]}")

    if BALANCE_MIN_COUNT > 0:
        print(f"开始补齐少数动作到 {BALANCE_MIN_COUNT} ...")
        X_input, y = balance_minor_actions(
            nn_model,
            X_input,
            y,
            min_count=BALANCE_MIN_COUNT,
            max_attempts=BALANCE_MAX_ATTEMPTS,
            obs_noise=BALANCE_OBS_NOISE,
            seed=42,
        )
        counts = np.bincount(y, minlength=NUM_ACTIONS)
        print(f"补齐后 action counts: {counts.tolist()}")
    if len(unique_classes) < 2:
        print("警告：当前数据只激活到单一动作，补充合成数据以覆盖更多动作")
        X_extra = get_train_data()
        y_extra = nn_predict(nn_model, X_extra)
        X_input = np.vstack([X_input, X_extra])
        y = np.concatenate([y, y_extra])
        if weights is not None:
            w_extra = np.ones(len(X_extra), dtype=np.float32)
            weights = np.concatenate([weights, w_extra])
        unique_classes = np.unique(y)
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

    if USE_CLASS_WEIGHTS:
        class_counts = np.bincount(y_tr, minlength=NUM_ACTIONS).astype(np.float32)
        class_counts[class_counts == 0] = 1.0
        class_weights = class_counts.sum() / (NUM_ACTIONS * class_counts)
        sample_w = class_weights[y_tr]
        w_tr = sample_w if w_tr is None else (w_tr * sample_w)

    # 配置CatBoost（分类器/回归器）
    print("进入蒸馏阶段：训练CatBoost")
    if DISTILL_USE_PROBS:
        cb_params = {'loss_function': 'MultiRMSE',
                     'early_stopping_rounds': 50,
                     'verbose': 1,
                     'max_depth': 9,
                     'learning_rate': 1,
                     'iterations': 16,
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
                     'max_depth': 7,
                     'learning_rate': 1,
                     'iterations': 16,
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
        y_pred_tr = np.array(cbr.predict(X_tr)).flatten()
        y_tr_flat = np.array(y_tr).flatten()
        acc_tr = (np.round(y_pred_tr).astype(int) == y_tr_flat).mean()
        print(f"Train Accuracy: {acc_tr}")

    # 保存蒸馏后的CatBoost模型到文件，便于部署和查表化
    print("保存蒸馏模型")
    cbr.save_model('distilled_catboost_model.cbm')
    print("CatBoost模型已保存为 distilled_catboost_model.cbm")

    # 保存训练数据归档
    try:
        archive = np.hstack([X_input, y.reshape(-1, 1).astype(np.float32)])
        np.savetxt(ARCHIVE_PATH, archive, delimiter=',', fmt='%.6f')
        print(f"训练数据已归档: {ARCHIVE_PATH}")
    except Exception as e:
        print(f"保存归档失败: {e}")

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

    if EVAL_RANDOM_SAMPLES > 0:
        print(f"随机抽样评估: {EVAL_RANDOM_SAMPLES} 条")
        rng = np.random.default_rng(EVAL_RANDOM_SEED)
        sample_idx = rng.choice(len(X_input), size=min(EVAL_RANDOM_SAMPLES, len(X_input)), replace=False)
        for i in sample_idx:
            state = X_input[i]
            cb_pred = cbr.predict([state])
            nn_pred = nn_predict(nn_model, [state])
            if DISTILL_USE_PROBS:
                cb_idx = int(np.argmax(cb_pred[0]))
            else:
                cb_idx = int(np.round(cb_pred[0]))
            nn_idx = int(nn_pred[0])
            print(f"state={state.tolist()} | distilled idx: {cb_idx}, value: {ACTION_WEIGHTS[cb_idx]} | nn idx: {nn_idx}, value: {ACTION_WEIGHTS[nn_idx]}")

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
