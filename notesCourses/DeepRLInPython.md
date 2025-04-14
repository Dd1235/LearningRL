# Deep RL in Python.

Ch1. Introduction to DRL
Ch2. Deep Q-Learning
Ch3. Introduction to Policy Gradient Methods
Ch4. Proximal Policy Optimization and DRL Tips

# Introdction to DRL

## Setting up the Environment


```python
env = gym.make("ALE/SpaceInvaders-v5")

class Network(nn.Module):
    def __init__(self, dim_inputs, dim_outputs):
        super(Network, self).__init__()
        self.linear = nn.Linear(dim_inputs, dim_outputs)
    def forward(self, x):
        return self.linear(x)
network = Network(dim_inputs, dim_outputs)
```
## Basic Loop


```python
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    while not done:
        action = select_action(network, state)
        next_state,  reward, terminated, truncated, _ = (env.step(action))
        done = terminated or truncated
        loss = calculate_loss(network, state, action, next_state , reward, done)
        optimizer.zero_grad()
        loss
        optimizer.step()
        state = next_state
```


# to be added from slides

