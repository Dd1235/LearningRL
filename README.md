# LearningRL

My attempt at teaching myself some Reinforcement Learning, and documenting stuff along the way, so as to help with revision, better my understanding, and maybe help others too(hopefully).

## Table of Contents

- [Introduction](SBnotes/Introduction.md)
- [Ch2 Multi Armed Bandits](SBnotes/MultiArmedBandits.md)
- [Ch3 Finite MDP](SBnotes/FMDP.md)
- [Ch4 DP](SBnotes/DP.md)
- [overview of RL methods](SBnotes/OverviewMethods.md)
- [Ch5 5.1 to 5.4](SBnotes/5.1TO5.4.pdf)
- [some misc notes on TD and Q](SBnotes/TD_Q_SARSA.md)


PS the markdown might not look right in the github preview for formuale, but works with any markdown previewer in VSCode or any other md previewer.

There are also some misc notes in the SBnotes/ folder.

## Codes

- [Bandits](code/Bandits)

    - Implementation of Multi armed bandits problem, using epison greedy, and upper confidence bound action method, and sample average and constant step estimators. Both stationary(normal distirbution) and non-stationary environments simulated.
    - added Median elimination with stationary distribution(simple case with std = 1 for all arms)

- [TicTactoe](code/TicTacToe)
    - using TD(0)
