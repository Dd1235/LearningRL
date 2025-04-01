import numpy as np


class FrozenLakeEnv:
    """
    A simple 4x4 'Frozen Lake' style environment, with:
      - Some 'hole' states (H) that are terminal with reward 0
      - A single 'goal' state (G) that is terminal with reward 1
      - A start state (S) at position (0,0) -> index 0
      - Other safe states '.' have no immediate reward

    States:
       Index in row-major order:
         0   1   2   3
         4   5   6   7
         8   9  10  11
        12  13  14  15

    Actions (integer):
      0 = Left
      1 = Down
      2 = Right
      3 = Up
    Config:
        S . . .
        . H . H
        . . . H
        H . . G

    """

    def __init__(self):
        self.nrow = 4
        self.ncol = 4
        self.nS = self.nrow * self.ncol  # total states
        self.nA = 4  # total actions

        # Predefined positions for holes and goal
        self.holes = {5, 7, 11, 12}
        self.goal = 15
        self.start_state = 0  # always start at state 0

    def in_bounds(self, r, c):
        return 0 <= r < self.nrow and 0 <= c < self.ncol

    def to_row_col(self, s):
        """Convert state index -> (row, col)."""
        return divmod(s, self.ncol)

    def to_s(self, r, c):
        """Convert (row, col) -> state index."""
        return r * self.ncol + c

    def step(self, state, action):
        """
        Applies a deterministic step from 'state' using 'action',
        returning (next_state, reward, done).
        """
        # If already in terminal state, just remain there
        if state in self.holes or state == self.goal:
            return state, 0.0, True

        r, c = self.to_row_col(state)
        if action == 0:  # left
            c -= 1
        elif action == 1:  # down
            r += 1
        elif action == 2:  # right
            c += 1
        elif action == 3:  # up
            r -= 1

        # If out-of-bounds, stay in the same cell
        if not self.in_bounds(r, c):
            next_state = state
        else:
            next_state = self.to_s(r, c)

        if next_state in self.holes:
            # Fell into a hole -> 0 reward, terminal
            return next_state, 0.0, True
        elif next_state == self.goal:
            # Reached goal -> reward=1, terminal
            return next_state, 1.0, True
        else:
            # Regular frozen cell -> 0 reward, non-terminal
            return next_state, 0.0, False


def print_environment_layout(env):

    layout = []
    for row in range(env.nrow):
        row_cells = []
        for col in range(env.ncol):
            s = env.to_s(row, col)
            if s == env.start_state:
                row_cells.append("S")
            elif s in env.holes:
                row_cells.append("H")
            elif s == env.goal:
                row_cells.append("G")
            else:
                row_cells.append(".")
        layout.append(" ".join(row_cells))
    print("Environment Layout:")
    for row_str in layout:
        print("  " + row_str)


def evaluate_policy(env, policy, gamma=0.9, theta=1e-8):
    """
    Given a policy (either deterministic or stochastic),
    iteratively evaluate it to find state-values V(s).

    - policy[s] can be:
        1. A single integer 'a' (deterministic), or
        2. An array-like of length env.nA that sums to 1 (stochastic).

    Returns:
      V: a 1D numpy array of shape (env.nS,) for the value of each state.
    """
    V = np.zeros(env.nS)

    while True:
        delta = 0.0
        for s in range(env.nS):
            # Terminal states (holes or goal) have zero value
            if s in env.holes or s == env.goal:
                V[s] = 0.0
                continue

            v_old = V[s]

            # 1) Identify the actions and their probabilities under this policy
            #    Note: we must consider np.integer as int as well.
            if isinstance(policy[s], (int, np.integer)):
                # Deterministic policy
                actions = [int(policy[s])]
                probs = [1.0]
            else:
                # Stochastic policy
                actions = list(range(env.nA))
                probs = policy[s]
                # Debug checks
                if len(probs) != env.nA:
                    raise ValueError(
                        f"Policy at state {s} has length {len(probs)} instead of {env.nA}."
                    )
                if not np.isclose(sum(probs), 1.0):
                    raise ValueError(
                        f"Policy at state {s} does not sum to 1. Got sum={sum(probs)}."
                    )

            # 2) Compute new value by summing over all (a, p_a)
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
    """
    Given a state-value function V, choose for each state s the action a
    that maximizes Q(s,a) = r + gamma * V[s'].
    Returns a deterministic policy (integer action per state).
    """
    new_policy = np.zeros(env.nS, dtype=int)

    for s in range(env.nS):
        if s in env.holes or s == env.goal:
            new_policy[s] = 0  # arbitrary, won't be used
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
    """
    Policy Iteration:
      1) Initialize a policy (e.g., all zeros).
      2) Policy Evaluation -> get V(s).
      3) Policy Improvement -> get new policy.
      4) Repeat until policy is stable.
    Returns final (policy, V, iteration_count).
    """
    policy = np.zeros(env.nS, dtype=int)  # deterministic: action=0 for all states
    iteration = 0

    while True:
        iteration += 1
        V = evaluate_policy(env, policy, gamma, theta)
        new_policy = greedy_policy_improvement(env, V, gamma)
        if np.all(new_policy == policy):
            break
        policy = new_policy

    return policy, V, iteration


def value_iteration(env, gamma=0.9, theta=1e-8):
    """
    Value Iteration:
      1) Start with V(s)=0 for all s.
      2) For each state s, update: V[s] = max_a [ r + gamma*V[next_s] ].
      3) Repeat until V converges (difference < theta).
      4) Greedy policy from the final V.

    Returns final (policy, V, iteration_count).
    """
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

            V[s] = max(q_values)
            delta = max(delta, abs(V[s] - v_old))

        if delta < theta:
            break

    # Derive a deterministic policy from the final V
    policy = np.zeros(env.nS, dtype=int)
    for s in range(env.nS):
        if s in env.holes or s == env.goal:
            policy[s] = 0
            continue

        # Compute best action
        q_values = []
        for a in range(env.nA):
            next_s, r, done = env.step(s, a)
            q_values.append(r + gamma * V[next_s] * (not done))
        policy[s] = np.argmax(q_values)

    return policy, V, iteration


def test_policy(env, policy, episodes=1000):
    """
    Test a deterministic policy by running 'episodes' from the start state.
    Returns the fraction of times we end up in the goal state.
    """
    success = 0
    for _ in range(episodes):
        s = env.start_state
        done = False
        while not done:
            # If policy is deterministic, policy[s] is an int or np.int64
            if isinstance(policy[s], (int, np.integer)):
                a = policy[s]
            else:
                # If it's stochastic, pick an action by distribution
                a = np.random.choice(range(env.nA), p=policy[s])
            s, r, done = env.step(s, a)
            if done and s == env.goal:
                success += 1
    return success / episodes


if __name__ == "__main__":
    env = FrozenLakeEnv()
    print_environment_layout(env)

    GAMMA = 0.9
    THETA = 1e-8

    print("\n[Policy Iteration]")
    pi_policy, pi_V, pi_iters = policy_iteration(env, gamma=GAMMA, theta=THETA)
    print(f"  Converged in {pi_iters} iterations.")
    print("  Policy (reshape as 4x4 for clarity):")
    print(pi_policy.reshape((4, 4)))
    print("  Value function V(s) (reshape as 4x4):")
    print(pi_V.reshape((4, 4)))
    success_rate_pi = test_policy(env, pi_policy, episodes=1000)
    print(f"  Success Rate over 1000 trials = {success_rate_pi:.2f}")

    print("\n[Value Iteration]")
    vi_policy, vi_V, vi_iters = value_iteration(env, gamma=GAMMA, theta=THETA)
    print(f"  Converged in {vi_iters} iterations.")
    print("  Policy (reshape as 4x4 for clarity):")
    print(vi_policy.reshape((4, 4)))
    print("  Value function V(s) (reshape as 4x4):")
    print(vi_V.reshape((4, 4)))
    success_rate_vi = test_policy(env, vi_policy, episodes=1000)
    print(f"  Success Rate over 1000 trials = {success_rate_vi:.2f}")
