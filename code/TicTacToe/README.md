# TIC TAC TOE using RL

Attempt at implementing a simple TIC TAC TOE agent 
Haven't reached chapter 6 of SB yet, but wanted to make something

AI is pretty bad though, opponent makes random moves, estimating V(s) and not Q(s,a), and only terminal states give reward, could maybe penalizing making moves, giving -0.1 for each move for faster win, or maybe +0.5 for draw etc.

## Theory

State value function
$$
V_{\pi}(s) = \mathbf{E}_{\pi}[\frac{G_t}{S_t = s}]
$$
We use the update rule:

### 🧾 TD(0) Update Rule:

\[
V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]
\]

Where:

- \( \alpha \): learning rate
- \( \gamma \): discount factor (your implementation uses \( \gamma = 1 \))
- \( R_{t+1} \): reward after taking action
- \( S_t \): current state
- \( S_{t+1} \): next state
- \( V(S_t) \): current value estimate

Essentially `mdp[board] += alpha * (mdp[next_state] - mdp[board])`
R_{t+1} is 0 for non-terminal

(This is different from MC where in you wait for the episode to finish before updating, here we are boot strapping)



## Usage

`python trainer.py`

This will:
- simulate 10,000 games using $ \epsilon$-greedy TD learning
- update the dictionary of state values (MDP)
- save it to mdp_values.pkl

pkl : Python pickle file that stores the dictionary mapping, board_state_string -> estimated values

`python play.py`
- AI plays as X , you play as O
- You enter your move as `row col`

