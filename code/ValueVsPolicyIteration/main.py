import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class FrozenLakeEnv:
    """
    0 1 2 3
    4 5 6 7
    8 9 10 11
    12 13 14 15

    or

    S .  . .
    . H . H
    . . . H
    H . . T
    """

    def __init__(self):
        self.nrow = 4
        self.ncol = 4
        self.nS = self.nrow * self.ncol  # number of states
        self.nA = 4  # actions, right left up down
        self.holes = {5, 7, 11, 12}
        self.goal = 15
        self.start_state = 0

    def in_bounds(self, r, c):
        return 0 <= r < self.nrow and 0 <= c < self.ncol

    def to_row_col(self, s):
        return divmod(s, self.ncol)

    def to_s(self, r, c):
        return r * self.ncol + c

    def step(self, state, action):
        """
        From state s, take action a, return (next state, reward, is Terminal)
        Here deterministic policy
        """
        if state in self.holes or state == self.goal:
            return state, 0.0, True
        r, c = self.to_row_col(state)
        if action == 0:
            c -= 1
        elif action == 1:
            r += 1
        elif action == 2:
            c += 1
        elif action == 3:
            r -= 1
        if not self.in_bounds(r, c):
            next_state = state
        else:
            next_state = self.to_s(r, c)
        if next_state in self.holes:
            return next_state, 0.0, True
        elif next_state == self.goal:
            return next_state, 1.0, True
        else:
            return next_state, 0.0, False


def evaluate_policy(env, policy, gamma=0.9, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0.0
        for s in range(env.nS):
            if s in env.holes or s == env.goal:
                V[s] = 0.0
                continue
            v_old = V[s]
            if isinstance(policy[s], (int, np.integer)):  # for deterministic policy
                actions = [int(policy[s])]
                probs = [1.0]
            else:
                actions = list(range(env.nA))  # non deterministic
                probs = policy[s]
            v_new = 0.0
            for a, p_a in zip(actions, probs):
                next_s, r, done = env.step(s, a)
                v_new += p_a * (r + gamma * V[next_s] * (not done))
            V[s] = v_new
            delta = max(delta, abs(v_new - v_old))
        if delta < theta:
            break
    return V


def greedy_policy_improvement(env, V, gamma=0.9):
    new_policy = np.zeros(env.nS, dtype=int)
    for s in range(env.nS):
        if s in env.holes or s == env.goal:
            new_policy[s] = 0
            continue
        best_value = -float("inf")
        best_action = 0
        for a in range(env.nA):
            next_s, r, done = env.step(s, a)
            q_sa = r + gamma * V[next_s] * (not done)
            if q_sa > best_value:
                best_value = q_sa
                best_action = a
        new_policy[s] = best_action
    return new_policy


def policy_iteration(env, gamma=0.9, theta=1e-8):
    policy = np.zeros(env.nS, dtype=int)
    iteration = 0
    while True:
        iteration += 1
        V = evaluate_policy(env, policy, gamma, theta)  # evaluate policy
        new_policy = greedy_policy_improvement(env, V, gamma)  # improve it
        if np.all(new_policy == policy):  # policy did not improve, stop
            break
        policy = new_policy
    return (
        policy,
        V,
        iteration,
    )  # policy, V, num of (evaluate, improve) steps done in policy iteration


def value_iteration(env, gamma=0.9, theta=1e-8):
    V = np.zeros(env.nS)
    iteration = 0
    while True:
        iteration += 1
        delta = 0.0
        for s in range(env.nS):
            if s in env.holes or s == env.goal:
                V[s] = 0.0
                continue
            v_old = V[s]
            q_values = []
            for a in range(env.nA):
                next_s, r, done = env.step(s, a)
                q_values.append(r + gamma * V[next_s] * (not done))
            V[s] = max(q_values)  # assign the maximum q value found
            delta = max(delta, abs(V[s] - v_old))
        if delta < theta:
            break
    policy = np.zeros(env.nS, dtype=int)
    for s in range(env.nS):
        if s in env.holes or s == env.goal:
            policy[s] = 0
            continue
        q_values = []
        for a in range(env.nA):
            next_s, r, done = env.step(s, a)
            q_values.append(r + gamma * V[next_s] * (not done))
        policy[s] = np.argmax(q_values)
    return policy, V, iteration


def test_policy(env, policy, episodes=1000):
    success = 0
    for _ in range(episodes):
        s = env.start_state
        done = False
        while not done:
            a = (
                policy[s]
                if isinstance(policy[s], (int, np.integer))
                else np.random.choice(range(env.nA), p=policy[s])
            )
            s, r, done = env.step(s, a)
            if done and s == env.goal:
                success += 1
    return success / episodes


def plot_policy(policy, title, env, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.set_xlim(0, env.ncol)
    ax.set_ylim(0, env.nrow)
    ax.set_xticks(np.arange(0, env.ncol + 1))
    ax.set_yticks(np.arange(0, env.nrow + 1))
    ax.grid(True)
    direction_map = {0: (-0.3, 0), 1: (0, -0.3), 2: (0.3, 0), 3: (0, 0.3)}
    for s in range(env.nS):
        r, c = env.to_row_col(s)
        y = env.nrow - r - 1
        x = c
        if s in env.holes:
            ax.add_patch(patches.Rectangle((x, y), 1, 1, color="black"))
            ax.text(
                x + 0.5,
                y + 0.5,
                "H",
                ha="center",
                va="center",
                color="white",
                fontsize=14,
            )
        elif s == env.goal:
            ax.add_patch(patches.Rectangle((x, y), 1, 1, color="green"))
            ax.text(
                x + 0.5,
                y + 0.5,
                "G",
                ha="center",
                va="center",
                color="white",
                fontsize=14,
            )
        elif s == env.start_state:
            ax.add_patch(patches.Rectangle((x, y), 1, 1, color="blue"))
            ax.text(
                x + 0.5,
                y + 0.5,
                "S",
                ha="center",
                va="center",
                color="white",
                fontsize=14,
            )
        else:
            ax.add_patch(patches.Rectangle((x, y), 1, 1, fill=False))
            dx, dy = direction_map[policy[s]]
            ax.arrow(
                x + 0.5,
                y + 0.5,
                dx,
                dy,
                head_width=0.2,
                head_length=0.2,
                fc="red",
                ec="red",
            )
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    env = FrozenLakeEnv()
    print("\n[Policy Iteration]")
    start = time.time()
    pi_policy, pi_V, pi_iters = policy_iteration(env)
    pi_time = time.time() - start
    print(f"  Converged in {pi_iters} iterations.")
    print(f"  Time taken = {pi_time:.6f} seconds.")
    print("  Policy (reshape as 4x4 for clarity):")
    print(pi_policy.reshape((4, 4)))
    print("  Value function V(s) (reshape as 4x4):")
    print(np.round(pi_V.reshape((4, 4)), 3))
    success_rate_pi = test_policy(env, pi_policy)
    print(f"  Success Rate over 1000 trials = {success_rate_pi:.2f}")
    plot_policy(
        pi_policy, "Policy Iteration Policy", env, "policy_iteration_policy.png"
    )

    print("\n[Value Iteration]")
    start = time.time()
    vi_policy, vi_V, vi_iters = value_iteration(env)
    vi_time = time.time() - start
    print(f"  Converged in {vi_iters} iterations.")
    print(f"  Time taken = {vi_time:.6f} seconds.")
    print("  Policy (reshape as 4x4 for clarity):")
    print(vi_policy.reshape((4, 4)))
    print("  Value function V(s) (reshape as 4x4):")
    print(np.round(vi_V.reshape((4, 4)), 3))
    success_rate_vi = test_policy(env, vi_policy)
    print(f"  Success Rate over 1000 trials = {success_rate_vi:.2f}")
    plot_policy(vi_policy, "Value Iteration Policy", env, "value_iteration_policy.png")
