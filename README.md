# Prism & LUT-Prism: Lightweight RL-based Congestion Control

Prism and LUT-Prism present a lightweight and deployment-oriented framework for Reinforcement Learning (RL)-based congestion control. Instead of directly deploying computationally intensive neural network policies, the framework distills trained RL models into efficient and interpretable representations, including gradient-boosted decision trees (CatBoost) and discrete lookup tables (LUTs).

The key motivation is to bridge the gap between high-performance learning-based control and the stringent latency, memory, and programmability constraints of real-world network systems. By transforming neural policies into structured models, Prism enables line-rate inference, hardware-friendly execution, and practical deployment in high-speed environments such as programmable NICs and network simulators.

##  Key Features

- **High Efficiency**: Achieves significantly faster inference compared to original neural network models.
- **Hardware-Friendly Design**: Models can be transformed into simple decision logic or table-based representations.
- **Hybrid Training Strategy**: Combines real-world trajectories with synthetic data to improve robustness.
- **Flexible Distillation Targets**: Supports both action classification and probability distribution-based distillation.

---

## Core Components

- **`distill_network.py`**  
  Distills reinforcement learning checkpoints into decision tree ensemble models (CatBoost format).

- **`distill_lut.py`**  
  Converts continuous state space into a discretized lookup table representation.

- **`PPO_AICC/`**  
  Contains integration logic for simulation environments and closed-loop evaluation pipelines.

- **Visualization Tools**  
  Includes scripts for analyzing and interpreting distilled models.

---

## Evaluation

The effectiveness of the distillation framework is evaluated using:

1. **Model Consistency**  
   Measures alignment between the distilled model and the original neural network.

2. **Control Performance**  
   Evaluates behavior under different network conditions, including:
   - Under-utilized states  
   - Target operating states  
   - Congested states  

3. **End-to-End Performance**  
   Assessed through closed-loop testing within network simulation environments.

---

## Notes

- This project is currently under active development.
- Full reproducibility and deployment instructions will be released in future updates.

---
