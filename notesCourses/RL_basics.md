Reinforcement Learning with Gymnasium in Python

# Outline

- Introduction to RL
- Model Based Learning
- Model Free Learning
- Advanced Strategies in Model-Free RL

# Ch 1Introduction to Reinforcement Learning

||Supervised|Unsupervised|Reinforcement Learning|
|---|---|---|---|
|Data Type|Labeled data|Unlabeled data| No predefined training data|
|Objective|Predict outcomes|Discover underlying patterns| Make decisions that maximize reward from the environment|
|Suitability|Classification, Regression|Clustering, Association|Decision-making tasks|


## When to use RL?

- sequential decision making, decisions influence future observations
- learning through rewards and penalties
    -  no direct supervision

Appropriate for RL: playing video games
Inappropriate: in-game object recognition

## Applications

- robotics
- finance, optimizing trade and investment, maximise profit
- autonomous vehicles, autonomous vehicles, enhancing safety and efficiency, minimizing accident risks
- chatbot development

## Navigating the RL framework


Agent
State Action reward
environment

### Interaction loop

```Python
env = create_environment()
state = env.get_initial_state()

for i in range(n_iterations):
    action = choose_action(state)
    state, reward = env.execute(action)
    update_knowledge(state, aciton, reward)
```

### Episodic vs. Continuous tasks

- Episodic tasks
    - tasks segment in episodes
    - episode has beginning and end
    - eg: agent playing chess

- Continuous tasks
    - continous Interaction
    - no distinct episoddes
    - eg: robot navigating a room, adjusting traffic lights

### Return

- sum of expected rewards = $ r_1 + r_2 + r_3 + ... + r_n $

### Discounted Return

- gives more weight to nearer rewards
- $ G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... $

\[
G = r_1 + \gamma r_2 + \gamma^2 r_3 + \gamma^3 r_4 + \cdots + \gamma^{n-1} r_n
\]

$\gamma$ is the discount factor

- Between zero and one
- Balances immediate vs long term rewards
- lower values -> immediate gains
- higher values -> long term gains

```Python
expected_rewards = np.array([1, 6, 3])
discount_factor = 0.9
discounts = np.array([discount_factor**8 for i in range(len(expected_rewards))])
discounted_return = np.sum(expected_rewards*discounts)
```

## Interacting with Gymnasium environments

- standard library for RL tasks
- key gymnasium environments:
    - CartPole
    - MountainCar
    - FrozenLake
    - Taxi

### Creating and initializing the environment

```Python
import gymnasium as gym

env = gym.make('CartPole', render_mode = 'rgb_array')
state, info = env.reset(seed = 42)
print(state)
```

### Visualizing the state

```Python
import matplotlib.pyplot as plt

state_image = env.render()
plt.imshow(state_image)
plt.show()

def render():
    state_image = env.render()
    plt.imshow(state_image)
    plt.show()

render()
```

### Performing Applications

- 0: moving left
- 1: moving right

```Python
action = 1
state, reward, terminated, truncated, info = env.step(action)
```

### Interaction loops

```Python
while not terminated:
    action = 1
    state, reward, terminated, _, _ = env.step(action)
    render()
```
# Ch2 Model Based Learning

## Markov Decision Process

- models RL environments mathematically

complex env -> (States, actions, rewards, transition probabilities) MDP -> solve with model-based RL techniques

### Markov Property

- future state depends only on current state and aciton
- eg Frozen lake as MDP
    - states Positions agent can occupy
    - terminal states: lead to episode termination, holes or the goal
    - actoins: up, down, left, right
    - transitions: actions don't necessairly lead to desired state, stochastic transitions, move right may lead to left, down, no movement etc.
    - rewards: only given in goal state

```Python
import gymnasium as gym

env = gym.make('FrozenLake', is_slippery=True)
print(env.action_space)
print(env.observation_space)
print(env.action_space.n) # number of actions
print(env.observation_space.n) # number of states
print(env.unwrapped.P[state][action])
```

## Policies and State-value functions

- RL objective -> formulate optimal policy
- specify which action to take each state to maximize Return


### Grid world policy example

```Python

# 0: left, 1: down, 2: right, 3: up
policy = {
    0:1, 1:2, 2:1, # state : action
    3:1, 4:3, 5:1,
    6:2, 7:3 # at 5 you down down to 8 and reach goal
}

state, info = env.reset()
terminated = False

while not terminated:
    action = policy[state]
    state, reward, terminated, _, _ = env.step(action)
    render()
```

### State-value functions

- estimate the states worth
- $ V(s) = E[G_t | S_t = s] $

\[
V(s) = r_{s+1} + \gamma r_{s+2} + \gamma^2 r_{s+3} + \cdots + \gamma^{n-1} r_{s+n}
\]

- **State — value of \(s\)**: The expected sum of discounted rewards.
- **Explanation**:
  - Starting in state \(s\).
  - Following the policy.

V(goal state) = 0

### Bellman Equation

\[
V(s) = r_{s+1} + \gamma V(s+1)
\]

- State value of s is the sum of immediate reward and discounted value of the next state

### Computing state values

```Python
def compute_state_value(state):
    if state == terminal.state:
        return 0
    action = policy[state]
    _, next_state, reward, _  = env.unwrapped.P[state][action][0]
    return reward + gamma * compute_state_value(next_state)

terminal_state = 8
gamma = 1
V = {state: compute_state_value(state) for state in range(env.observation_space.n)}
print(V)
```

### Changing Policies

```Python
policy_two = {0:2, 1:2, 2:1,3:2, 4:2, 5:1,6:2, 7:2}
V_2 = {state: compute_state_value(state) for state inrange(num_states)}
print(V_2)
```

## Action-value functions

### Action-value functions (Q-values)

- Expected return of:
    - starting at as state s
    - taking action a
    - following the policy

- estimates desirability of actions within states

\[
    Q(s, a) = r_{a} + \gamma V(s+1)
\]

### Computing Q-values

```Python
def compute_q_value(state, action):
    if state == terminal_state:
        return None
    _, next_state, reward, _ =  env.unwrapped.P[state][action][0]
    return reward + gamme*compute_state_value(next_state)

Q = {(state, action): compute_q_value(state, action) for state in range(num_states) for action in range(num_actions)}
```


### Improving the policy

- selection for each state the action with the highest q value

```Python
improved_policy = {}

for state in range(num_states-1):
    max_action = max(range(num_actions), key = lambda action: Q[(state, action)])
    improved_policy[state] = max_action
```

## Policy iteration and value iteration

### Policy iteration

- iterative process to find optimal policy

- intialize policy -> (evaluate policy <-> improve policy) (repeat till policy stops changing) -> optimal policy

### Example code: Policy evaluation 
```Python
def policy_evaluation(policy):
    V = {state: compute_state_value(state, policy) for state in range(num_states)}
    return V
```

### Example code: Policy improvement

```Python
def policy_improvement(policy):
    improved_policy = {
        s: 0 for s in range(num_states-1)
    }
    Q = {(state_action): compute_q_value(state, action, policy)
    for state in range(num_states) for action in range(num_actions)}

    for state in range(num_states-1):
        max_action = max(range(num_actions), key = lambda action: Q[(state, action)])
        improved_policy[state] = max_action

    return improved_policy
```

### Example code: Policy iteration

```Python

def policy_iteration():
    policy = {0:1, 1:2, 2:1, 3:1, 4:3, 5:1, 6:2, 7:3}
    while True:
        V = policy_evaluation(policy)
        new_policy = policy_improvement(policy)
        if new_policy == policy:
            break
        policy = new_policy
    return policy, V

policy, V = policy_iteration()
```

### Value iteration

- combines policy evaluation and improvement in one step
    - computes optimal state-value functions
    - derives policy from it

```Python
V = {state: 0 for state in range(num_states)}
policy = {state:0 for state in range(num_states-1)}
threshold = 0.001

while True:
    new_V = {state: 0 for state in range(num_states)}
    for state in range(num_states-1):
        max_action, max_q_value = get_max_action_and_value(state, V)
        new_V[state] = max_q_value
        policy[state] = max_action

    if all(abs(new_V[state] = V[state]) < thresh for state in V):
        break
    V = new_V


def get_max_action_and_value(state, V):
    Q_values - [compute_q_value(state, action, V) for action in range(num_actions)]
    max_action = max(range(num_actions), key = lambda a: Q_values[a])
    max_q_value = Q_values[max_action]
    return max_action, max_q_value

def compute_q_value(state, action, V):
    if state == terminal_state:
        return None
    _, next_state, reward, _ = env.unwrapped.P[state][action][0]
    return reward + gamma*V[next_state]
```



# Ch 3 Model Free Learning

## Monte Carlo methods


MOdel-based learning:
- rely on knowledge of environment dynamics
- no interaction with environment

Model-free learning:

- doesn't realy on knowledge of environment dynamics
- agent interacts with environment
- learns policy through trial and error
- more suitable for real world Applications

- Monte Carlo methods
    - collect random episodes -> estimate A values using MC -> optimal policy


### Generating an episode


```Python
def generate_episode():
    episodes = []
    state, info = env.reset()
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated = env.step(action)
        episode.append([state, action, reward])
        state = next_state
    return episode
```

### First-visit Monte Carlo

```Python

def first_visit_mc(num_episodes):
    Q = np.zeros((num_states, num_action))
    returns_sum = np.zeros((num_states, num_actions))
    returns_count = np.zeros((num_states, num_actions))

    for i in range(num_episodes):
        episode = generate_episode()
        visited_states_action = set()
        for j, (state, action, reward) in enumerate(episode):
            if (state, action) not in visited_states:
                returns_sum[state, action] += sum([x[2] for x in episode[j:]])
                returns_count[state, action] += 1
                visited_states_actions.add((state, action))

    nonzero_counts = returns_count != 0
    Q[nonzero_counts] = returns_sum[nonzero_counts] / returns_count[nonzero_counts]

    return Q

```

### Entry-visit Monte Carlo

```Python
def every_visit_mc(num_episodes):
    Q = np.zeros((num_states, num_actions))
    returns_sum = np.zeros((num_states, num_actions))
    returns_count = np.zeros((num_states, num_actions))


    for i in range(num_episodes):
        episode = generate_episode()

        for j, (state, action, reward) += sum([x[2]] for x in episode[j:])
        returns_count[state, action] += 1

    nonzero_counts = returns_count != 0
    Q[nonzero_counts] = returns_sum[nonzero_counts] / returns_count[nonzero_counts]

    return Q
```

```Python
def get_policy():
    policy = {state: np.argmax(Q[state]) for state in range(num_states)}
    return policy
```

```Python
Q = first_value_mc(1000)
policy_first_value = get_policy()
Q = every_visit_mc(1000)
policy_every_first = get_policy()
```

## Temporal Difference learning


|TD Learning | Monte Carlo|
|---|---|
|Model free|Model free|
|estimate Q-table based on interaction|estimate Q-table based on interaction|
|update Q-table each step within episode|update Q-table when at least one episode done|
|suitable for tasks with long/indefinite episodes|suitable for short episodic tasks|

### SARSA

- curr state, curr action, collected reward, next state, next action

- **on policy method** : adjusts strategy based on taken actions

$$
Q(s, a) = (1 - \alpha)Q(s,a) + \alpha [r + y Q(s',a')]
$$
Where  (please please \( \) work for github inline math! 😭)
\(\alpha \) : learning rate
\(\gamma \) : discount factor

- initialization

```Python
env = gym.make("FrozenLake", is_slippery = False)
...
Q = np.zeros((num_states, num_actions))
alpha = 0.1
gamma = 1
num_episodes = 1000
```

- SARSA loop

```Python
for episode in range(num_episodes):
    state, info = env.reset()
    action = env.action_space.sample()
    terminated = False
    while not terminated:
        next_state, reward, termianted, truncated, info = env.step(action)
        next_action = env.action_space.sample()
        update_q_table(state, action ,reward, next_state, next_action)
        state, aciton = next_state, next_action

def update_q_table(state, action, reward, next_state, next_action):
    old_value = Q[state, action]
    next_value = Q[next_state, next_action]
    Q[state, action] = (1 - alpha)*old_value + alpha*(reward + gamma*next_value)

policy = get_policy()
```

## Q-Learning

- quality learning
- model-free technique

Intialize Q-table -> perform action -> get_reward -> update_q_table -> perform action

$$
Q(s,a) = (1-\alpha)Q(s,a) + \alpha[r + \gamma \text{max a'}Q(s', a')]
$$

|sarsa|q learning|
|updates based on action taken|independent|
|on-policy learner|off-policy learner|

```Python
env = gym.make("FrozenLake", is_slippery = True)

num_episodes = 1000
alpha = 0.1
gamma = 1

num_states, num_actions = env.observation_space.n, env.action_space.n
Q = np.zeos((num_states, num_actions))

reward_per_random_episode = []

for episode in range(num_episodes):
    state, info = env.reset()
    terminated = False
    episode_reward = 0
    while not terminated:
        action = env.action_space.sample()
        new_state, reward, terminated, truncated, info = env.step(action)
        update_q_table(state, action, new_state)
        episode_reward += reward
        state = new_state
    reward_per_random_episode.append(episode_reward)

def update_q_table(state, action, reward, new_state):
    old_value = Q[state, action]
    next_max = max(Q[new_state])
    Q[state,action] = (1-alpha)*old_value + alpha * (reward + gamma*next_max)

reward_per_learned_episode = []

policy = get_policy()
for episode in range(num_episodes):
    state, info = env.reset()
    terminated = False
    episode_reward = 0
    while not terminateed:
        action = policy[state]
        new_state, reward, terminated, truncated, info = env.step(action)
        state = new_state
        episode_reward += reward
    reward_per_learned_episode.append(episode_reward)
```
### Q learning evaluation

```Python
import numpy as np
import matplotlib.pyplot as plt

avg_random_reward = np.mean(reward_per_random_episode)
avg_learned_reward = np.mean(reward_per_learned_episode)

plt.bar(['Random Policy', 'Learned Policy'],
        [avg_random_reward, avg_learned_reward],
        color = ['blue', 'green'])

plt.title('Average reward per episode')
plt.ylabel('Average reward')
plt.show()
```

# ch 4 

## Expected SARSA

$$
Q(s,a) = (1-\alpha)Q(s,a) + \alpha(r + \gamma E{Q(s',A)})
$$

- takes into account all actions

E{Q(s',A)} = Sum(Prob(a) * Q(s',a) for a in A)

- random actions -> equal probabilities

E{Q(s',A)} = Mean(Q(s',a) for a in A)

```Python
env = gym.make('FrozenLake-v1', is_slippery=False)
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

gamma = 0.9
alpha= 0.1
num_episodes = 1000

def update_q_table(state, action, next_state, reward):
    expected_q = np.mean(Q[next_state])
    Q[state, action] = (1-alpha)*Q[state, action] + alpha * (reward + gamma*expected_q)

for i in range(num_episodes):
    state, info = env.reset()np
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        update_q_table(state, action, next_state, reward)
        state = next_state

policy = {state: np.argmax(Q[state]) for state in range(num_states)}
```

# Double Q-Learning

## Q learning
- overestimates Q-values by updating based on max Q
- might lead to suboptimal policy learning

## Double Q-learning

- maintains two Q-tables
- each table updated based on the other
- reduces risk of Q-values overestimation

- randomly select a table

$$
Q_0(s,a) = (1-\alpha)Q_0(s,a) + \alpha(r + \gamma Q_1(s', max_a))
$$


$$
Q_1(s,a) = (1-\alpha)Q_1(s,a) + \alpha(r + \gamma Q_0(s', max_a))
$$

- reduces overestimation bias
- alternates between Q_0 and Q_1 updates
- both tables contribute to learning process

```Python
env = gym.make('FrozenLake-v1', is_slippery = False)

num_states = env.observation_space.n
n_actions = env.action_space.n
Q = [np.zeros((num_states, num_actions)) ]*2

num_episodes = 1000
alpha = 0.5
gamma = 0.99

def update_q_tables(states, action, reward, nexxt_state):
    i = np.random.randint(2)
    best_next_action = np.argmax(Q[i][next_state])
    Q[i][state, action]= (1 - alpha)*Q[i][state, action] + alpha * (reward + gamma*Q[i-i][next_state, best_next_action])

for episodes in range(num_episodes):
    state, info = env.reset()
    termianted = False

    while not terminated:
        action = np.random.choice(n_actions)
        next_state, reward, terminated, truncated, info = env.step(action)
        update_q_tables(state, action, reward, next_state)
        state = next_state

final_Q = np.mean(Q, axis = 0) # either the average or
final_Q = Q[0] + Q[1] # sum of the two tables

policy = {state: np.argmax(final_Q[state]) for state in range(num_states)}
```

## Balancing exploration and exploitation


- till now with Temporal difference methods we have chosen actions randomly.
- prevents from optimizing strategy based on learned Q-values

```Python
env = gym.make('FrozenLake', is_slippery=True)

action_size = env.action_space.n
state_size = env.observation_space.n
Q = np.zeros((state_size, action_size))

alpha = 0.1
gamma = 0.99
total_episodes = 10_000

def epsilon_greedy(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        acton = np.argmax(Q[state,:]) # exploit

    return action

epsilon = 0.9
rewards_eps_greedy = []

for episode in range(total_episodes):
    state, info = env.reset()
    terminated = False
    episode_reward = 0
    while not terminateed:
        action = epsilon_greedy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        Q[state, action] = (1-alpha)*Q[state, action] + alpha*(reward + gamma*np.max(Q[next_state, :]))
        episode_reward += reward
        state = next_state
    rewards_eps_greedy.append(episode_reward)
```

Training decayed epsilon_greedy

```Python
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.999

rewards_decay_eps_greedy = []

for episode in range(total_episodes):
    state, info = env.reset()
    terminated = False
    episode_reward = 0
    while not terminated:
        action = epsilon_greedy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        Q[state, action] = (1-alpha)*Q[state, action] + alpha*(reward + gamma*np.max(Q[next_state, :]))
        episode_reward += reward
        state = next_state
    rewards_decay_eps_greedy.append(episode_reward)
    epsilon = max(min_epsilon, epsilon*epsilon_decay)
    epsilon = max(min_epsilon, epsilon*epsilon_decay)
```

### comparing Strategies

```Python
avg_eps_greedy = np.mean(rewards_eps_greedy)
avg_decay_eps_greedy = np.mean(rewards_decay_eps_greedy)
plt.bar(['Epsilon Greedy', 'Decayed Epsilon Greedy'],
        [avg_eps_greedy, avg_decay_eps_greedy],
        color = ['blue', 'green'])
plt.title('Average reward per episode')
plt.ylabel('Average reward')
plt.show()
```
### Multi-armed bandit problem

```Python
# Create a 10-armed bandit
true_bandit_probs, counts, values, rewards, selected_arms = create_multi_armed_bandit(10)

for i in range(n_iterations): 
  	# Select an arm
    arm = epsilon_greedy()
    # Compute the received reward
    reward = np.random.rand() < true_bandit_probs[arm]
    rewards[i] = reward
    selected_arms[i] = arm
    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]
    # Update epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Initialize the selection percentages with zeros
selections_percentage = np.zeros((n_iterations, n_bandits))
for i in range(n_iterations):
    selections_percentage[i, selected_arms[i]] = 1
# Compute the cumulative selection percentages 
selections_percentage = np.cumsum(selections_percentage, axis=0) / np.arange(1, n_iterations + 1).reshape(-1, 1)
for arm in range(n_bandits):
  	# Plot the cumulative selection percentage for each arm
    plt.plot(selections_percentage[:,arm], label=f'Bandit #{arm+1}')
plt.xlabel('Iteration Number')
plt.ylabel('Percentage of Bandit Selections (%)')
plt.legend()
plt.show()
for i, prob in enumerate(true_bandit_probs, 1):
    print(f"Bandit #{i} -> {prob:.2f}")

```