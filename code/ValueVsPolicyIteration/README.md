

# MDP, Policy Iteration, Value Iteration 

Formal description is in my notes [here](../SBnotes/FMDP.md)


Imagine you are going through a new city you moved into. There are states(locations) you can be in, say at the Museum or the Park, and there are actions you can take in a state, to move to a new state and get some reward. Say you take the metro from the Museum to the Park and receive some happiness +x.

Notice this is a special process, there is some probability of moving to state s' from a state s and action a, it is not dependedent on other history. And there is some reward on taking action a in state s. The reward I get from hopping on the metro from the Museum, is independent of whatever I was doing before.

So I can assign some value to each state, v*(s), which is the best long term reward I can expect from it (this reward calculation would involve some "discounting", wherein instead of adding up all the rewards as I move from this state to another to another, I give higher priority to rewards I receive earlier). I can also assign some value to each state, action pair, q(s,a), that say I am at the museum, and I hop on the metro, what is the best long term reward I can acheive from here? Get off at the Park and continue on collecting rewards from this state, or get off at the Art Gallery, and continue on your quest? Or you can have $ V^\pi(s) $ , this gives me the long term reward from my state s, given I do the action dictated by policy $\pi$.

And I can have a policy that tells me, hey, in this state, this action will lead to best long term rewards. Again, at any state, I have some action I should take(deterministic), or perhaps, some actions with probabilities, it doesn't matter how I got to this state, this is what makes it "Markov"


Now calculating this V and policy is not trivial. Afterall say there are n locations I can be in, and say the maximum number of actions possible at a state is k, then technically I have O(k^n) configurations possible. Think a tic tac toe game, I have 3^9 states for such a small game (albeit many states are invalid).

Say I have some policy Pi, it tells me something like You are at the Museum, so there is a 50% chance you take the metro,and
then a 70% you get of stop1, 30% at stop2. And that there is 1-50% = 50% chance I take a walk, and say 70% chance I walk to the ice cream shop,and a 30% chance I walk to the mall. (This process is stochastic since it gives me a probability I take an action from a state)

If my policy just gave me a 100% chance I take an action, like go on the metro, it would be deterministic. Where I get off after taking the metro is determined by the transition probabilities.

I can calculate the value of each state, under this policy, using Policy iteration. 
Over multiple iterations, in each iteration, I calculate the value of each state using the bellman equation, and this can be shown to converge mathematically.

Now I have evaluated this random policy, by doing policy iteration. How do I improve it? Policy evaluation. So given a policy, I need to output a new policy that is guaranteed to be a bit better. So you do this repeatedly to get the best policy ever.

(fact: there is always an optimal deterministic policy so it does suffice to just go over the deterministic policies)

So just act greedy. Let us say my current policy is to take the metro from the museum(deterministic), and that gives me Q(museum, metro) = 10,
and I know that Q(museum, bus) = 20, I just improve my policy to take the bus from the museum instead of the metro. So you just do this for every state we have, make a new policy greedily.

This was **Policy Evaluation**.

But the value function is now outdated for the new policy, so we need to do Policy iteration again to update it.
We just repeat this over and over again, till we get to the optimal policy (if your policy doesn't improve after you run this, you are at the optimal policy).

So if you take some random policy, calculate its V(s), improve it, calculate the new V(s), improve and continue this, this aglorithm would be **Policy Iteration**

In this algorithm, when we have a policy, it is quite some effort to calculate the value function that describes it perfectly. To implement it, you would probably do multiple iterations, each iteration, you update V(s) for each state, and do this till your value function isn't changing at all or not improving by that much.

What if you just stop after 5 iterations, just get some approximate idea of the value function? If my current policy is Pi(Museum) = Metro, and in reality Q(Museum, metro) = 10, but I do not "complete" the policy evaluatoin, so perhaps my Q(museum,metro) = 8, which is technically incorrect, but I start exploring here itself, and do policy evaluation to get a new policy. This again, can be mathematically shown to converge.

An extreme case is when you do policy evaluation (improve your policy), after just one iteration, only one update per state. This is called **value evaluation**.