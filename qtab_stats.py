#!/usr/bin/env python3
import sys
import numpy as np
from scipy.stats import entropy

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_qtable.py <qtable.npy>")
        sys.exit(1)

    path = sys.argv[1]
    q = np.load(path)

    if q.ndim != 2:
        raise ValueError(f"Expected a 2D Q-table (states × actions). Got shape {q.shape}")

    num_states, num_actions = q.shape
    print(f"Loaded Q-table: {num_states} states × {num_actions} actions\n")

    # ----------------------------------------
    # Basic statistics
    # ----------------------------------------
    print("=== GLOBAL STATISTICS ===")
    print(f"Min Q-value: {q.min():.4f}")
    print(f"Max Q-value: {q.max():.4f}")
    print(f"Mean Q-value: {q.mean():.4f}")
    print(f"Std  Q-value: {q.std():.4f}")
    print()

    # ----------------------------------------
    # Action preference analysis
    # ----------------------------------------
    greedy_actions = np.argmax(q, axis=1)
    action_counts = np.bincount(greedy_actions, minlength=num_actions)
    action_freqs = action_counts / num_states

    print("=== ACTION PREFERENCES (Greedy Policy) ===")
    for a in range(num_actions):
        print(f"Action {a}: chosen in {action_counts[a]} states ({action_freqs[a]*100:.2f}%)")

    dead_actions = [a for a, c in enumerate(action_counts) if c == 0]
    if dead_actions:
        print(f"\nDead actions (never greedy): {dead_actions}")
    print()

    # ----------------------------------------
    # Per-action Q-value statistics
    # ----------------------------------------
    print("=== PER-ACTION Q-VALUE STATS ===")
    for a in range(num_actions):
        print(f"Action {a}: mean={q[:, a].mean():.4f}, std={q[:, a].std():.4f}, "
              f"min={q[:, a].min():.4f}, max={q[:, a].max():.4f}")
    print()

    # ----------------------------------------
    # Policy entropy (how deterministic is the greedy policy?)
    # ----------------------------------------
    policy_entropy = entropy(action_freqs)  # natural log base
    max_entropy = np.log(num_actions)

    print("=== POLICY ENTROPY ===")
    print(f"Entropy of greedy policy: {policy_entropy:.4f} (max possible: {max_entropy:.4f})")
    print(f"Normalized entropy: {policy_entropy / max_entropy:.4f}")
    print()

    # ----------------------------------------
    # Information richness: Q-table dispersion and structure
    # ----------------------------------------
    # 1. Variance across actions for each state
    state_variances = q.var(axis=1)
    avg_variance = state_variances.mean()
    print("=== INFORMATION RICHNESS ===")
    print(f"Average variance across actions per state: {avg_variance:.4f}")

    # 2. Rank of flattened Q-table (linear independence)
    #    If Q-values are extremely structured, rank will be low.
    matrix_rank = np.linalg.matrix_rank(q)
    print(f"Numerical rank of Q-table matrix: {matrix_rank} / {min(num_states, num_actions)}")

    # 3. Normalized entropy per state of action distribution (softmax-like)
    def softmax(x):
        z = x - x.max()
        return np.exp(z) / np.exp(z).sum()

    state_entropies = np.array([entropy(softmax(q[s])) for s in range(num_states)])
    print(f"Average softmax-entropy across states: {state_entropies.mean():.4f} "
          f"(max={max_entropy:.4f})")
    print(f"Normalized softmax-entropy: {state_entropies.mean() / max_entropy:.4f}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
