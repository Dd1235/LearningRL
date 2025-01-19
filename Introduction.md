# Introduction

Reinforcement Learning is this sweet spot between control theory and machine learning.

It's a branch of machine learning that about learning control strategies for interacting with complex environments.


The word 'reinforce' comes from the fact that you reinforce good behaviours in Human/Animal subjects. So its a very biologically inspired idea.

You have an agent that interacts with the environment, it gets to measure its current state in the environment (this is not the same as having a full view of the environment, imagine a mouse in a maze, it sees the walls around but not a top view of the maze), and gets rewarded or punished based on the actions it takes. 

You pick how it gets rewarded. It isn't so straightforward. If you give a reward for every "correct" action, it is just the same as supervised learning (imagine you give the mouse a piece of cheese for every correct turn, that's just labelled data). So in some way it is semi supervised learning. Also imagine a game of chess, you have to measure how good a move is based on future outcomes, and even if you don't win, the move could've been very smart.


So this is very hard optimization problem.

The big challenge is to design on a policy $\pi$ , given a state s, that tells you what action, a, to take to maximize future rewards.

The environment is not deterministic but probabilitic. So the policy is also probabilitic.
(hence it isn't a 'control law' but a policy)

Policy: $π(s, a) = Pr(a = a \mid s = s)$

So part of designing a good policy is understanding what the value of being in a state s is.

Value 
$$
V_\pi(s) = \mathbb{E} \left[ \sum_t \gamma^t r_t \mid S_0 = s \right]
$$

$ \gamma^t $ is the discount rate, its between 0 and 1, and tells you how much you value future rewards. It its zero you only care about the immediate reward, if its 1 you care about all future rewards equally. 

(Question: why is it only between 0 and 1? Isn't it sometimes better to value future rewards more than immediate ones? Valuing immediate rewards more seems to be one of those very dumb human things. Reminds me of the experiment where kids are given a choice between having a chocholate now or waiting for 10 more minutes, and having two instead. Same for investors prioritizing short term gains over long term ones. Fretting over small losses during turbulances, instead of looking at the long term growth of the market. Shouldn't we try and remove this dumb human bias from our algorithms?

More thoughts: The questions actually seems dumb considering most rl problems deal with non stationary environments. If you're playing a game of chess, you don't know what the opponent will do, how could you 'measure' future rewards properly? Makes sense to prioritize immediate gains? Is this the right answer?)



We model the environment as a Markov Decision Process (MDP). There is a probability of going from my current state and action to another state. Makes it hard to optimize these policies.

## Credit Assignment problem

Central challenge in RL. 
Because your rewards are often very sparse and infrequent its very hard to tell what action sequence was responsible for the reward.

Identified in early 1960s by Minsky.

## some more stuff

---

### **Dense vs Sparse Rewards**
- **Dense Rewards**: Rewards are provided frequently, offering more immediate feedback for the agent's actions.
- **Sparse Rewards**: Rewards are rare or delayed, making it harder for the agent to learn the optimal policy.
  - Sparse rewards lead to **low sample efficiency**, requiring the agent to play many times to discover a good policy.

---

### **Reward Shaping**
- In systems with sparse rewards, **reward shaping** is often used.
- **Reward Shaping**: A human expert adds intermediate rewards to guide the agent toward the desired goal.
- Helps address the difficulty of sparse rewards but must be carefully designed to avoid introducing unintended biases.


### **Optimization in Reinforcement Learning**

Essentially we need to solve **hard, non-linear, and non-convex optimization problems**.
ML solves it using data.
Control theory solves it using models.

Different approaches include:

1. **Differential Programming**:
   - Uses gradients to optimize the policy or value functions.
   - Requires differentiable models.

2. **Monte Carlo Methods**:
   - Randomly try many possible solutions.
   - Simple but computationally expensive.

3. **Temporal Difference (TD) Learning**:
   - Strikes a balance between **differential programming** and **Monte Carlo** methods.
   - **Model-free**: Does not require a model of the system.
   - Learns by bootstrapping: Updating estimates based on other learned estimates.

4. **Bellman (1957)**:
   - Key pioneer of **optimal control theory**.
   - Bellman equations are central to RL for value function updates.



### **Exploration vs Exploitation**
- A fundamental challenge in **machine learning** and **reinforcement learning**.
- **Exploration**: Invest effort in trying new actions or policies to discover better strategies.
- **Exploitation**: Use the current best-known policy to maximize immediate rewards.
- Balancing these two is critical for achieving optimal learning.



### **Optimization Techniques**
- Various methods can be used to optimize policies or value functions in RL:
  - **Gradient Descent**
  - **Evolutionary Optimization**
  - **Simulated Annealing**
  - **Policy Iteration**: Iteratively improve the policy based on value function updates.




[cool video: Learning Motor Primitives for Robotics](https://www.youtube.com/watch?v=rbSFlLtnMFM)

[cool video: pilco learner](https://www.youtube.com/watch?v=XiigTGKZfks)

### Q learning

Instead of learning policy and value functions separately, you kind of learn them together.

$ Q(s,a)$ = quality of state/action pair 

something like the value function of state and action together assuming you do the best thing you can in the future.

$$
Q_{\text{update}}(s_t, a_t) = Q_{\text{old}}(s_t, a_t) + \alpha \left( r_t + \gamma \max_a Q(s_{t+1}, a) - Q_{\text{old}}(s_t, a_t) \right)
$$

where $\alpha$ is the learning rate.

### Hindsight Replay

Maybe my set of actions will be good for a different reward. Maybe someday I'll want to do this other thing. More like something a human would do. Make more sample efficient.