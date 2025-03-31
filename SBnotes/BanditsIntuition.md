# Multi Armed Bandits problem

Imagine you move to a new city, and your objective is to find your favourite restaurant, which you can visit everyday. There are multiple, say k restaurants in the city, and initially you know nothing about them.

Assume each restaurant has some "average/mean" happiness it gives you, with some unknown standard deviation. The underlying distribution could be anything, and each restaurants distribution is independent of the other, and assume it is stationary, that is it not changing over time for a particular restaurant.

Now the obvious solution is one wherein you visit each restaurant infinite number of times, the law of large numbers tells you that you will know each restaurants distribution and voila, you have your solution. This approach **asymptotically converges**, but we can probably do better than visit every restaurant an infinite number of times.

At any given point of time, you can choose to either pick the best restaurant you have seen till now, ie, exploit your knowledge, or explore, try out a new one or give a restaurant a second or third chance.

This makes intuitive sense, perhaps, you happened to visit a nice Malayali restaurant on Onam, and ended up waiting 1.5 hours in queue, and did not get great service because of the high rush, and received hence a "bad reward" that instance. Or you visited an okayish restaurant on its opening day and got free food, and were left feeling super giddy. Or perhaps you were just upset that day because your Manager gave you too much workload, so the "happiness/reward", you received wasn't so high even at a good restaurant. 

But it is inefficient to just keep exploring. At some point, you need to exploit what you have observed. Perhaps, one restaurant repeatedly gave you stomach aches, you should probably not visit it as often and go to more promising restaurants. 

This underscores the **tradeoff** between **exploration** and **exploitation**.

It is very much like the dilemmas we face in real life, **"hmm I seem to like Machine Learning and Databases/Backend a lot, do I exploit this knowledge and take all my electives related to these two topics? Or do I explore a bit, and take the Computer Graphics elective, perhaps I will enjoy it?"**

So let us say you have a policy wherein, with a probability of $ \epsilon $ you pick a restaurant randomly, and 1 - $ \epsilon $ times the best restaurant you have seen so far, that is follow a greedy strategy. But you need to converge to teh best restaurant at some point, so you need to decrease how much you explore and you know more and more, and **decay** this "exploration probability" each time step.

But there is a problem here, when you are exploring, it makes sense to explore your second favourite restaurant a bit more than one that made your stomach feel funny one too many times. 

So it makes sense to **weigh** the exploration, based on the happiness the restaurant has given you so far, call it its Q value.


$$
Q_t(a) = \frac{\text{sum of rewards when a taken before t}}{\text{number of times a taken before t}} = \frac{\sum_{i=1}^{t-1} R_i \cdot 1_{A_i = a}}{\sum_{i=1}^{t-1} 1_{A_i = a}}
$$

(we instead weight it using the softmax exploration, since Q values can be negative)

With this previous epsilon-greedy strategy we are not factoring in exploring "unexplored" restaurants more often.

Now lets try being a bit **optimistic** here. If I have explored a restaurant very little, it makes sense I explore it since I am **uncertain** of how it might be. 

This gives rise to the **UCB** strategy.

\[
\text{UCB}_i(t) = \hat{\mu}_i(t) + \sqrt{\frac{2\ln t}{n_i(t)}}
\]

- \(\hat{\mu}_i(t)\): Empirical mean reward of restaurant \(i\).
- \(n_i(t)\): Number of times arm \(i\) was visited by time \(t\).

We choose the restaurant that gives the maximum value for the above term at every time step.
The first term is the exploitative term, higher for restaurants that have given you better rewards till now. But we need to explore too. The second term is the "entropy/exploration" term.

The fewer times you have visited a restaurant by time t, the larger will the entropy term be, increasing the restaurant's chances of being explored. This also takes care of making sure we do not "under/over explore" a restaurant because of one very good or bad experience, because means get skewed by outliers. 

Over time Bad restaurants get eliminated as both the terms stay low. While good restaurants get visited more and more often.
Think of all the times initialily you received much lower happiness than what the best restaurant gives you, you "regret" it, and regret is zero when you visit your favourite restaurant. Regret, in this context, is the sum total over all time steps of the difference between the best value and the value you received at each time step.

UCB is a "regret optimal" algorithm. UCB ensures that cumulative regret grows only logarithmically with time, meaning your average regret per time step shrinks over time. This can be mathematically proven.

Comparing UCB and Epsilon Greedy:

- UCB takes time to kick in. Initially it would be mostly exploration, more since UCB would be implemented such that each restaurant is visited once, before even using the formula(to avoid division by zero error).
- so in a smaller time frame UCB might perform worse, it is difficult to say certainly, since environment is stochastic. But think 15 restaurants, 7 days. Epsilon greedy might be better.
- it is also more computationally expensive
- and given the mean term, UCB balance evidence and optimism, if you get food poisonding from your favourite restaurant on day 1, the first term is going to be really low, and it will take time for UCB to "forgive" that restaurant.
- but its certianly better over a long term horizon

Neither of these approaches seem to be eliminating restaurants for good. Practically speaking, if a restaurant gave you a stomach bug each time you visited it, it probably isn't the best one. But you cannot be sure.

So assume an approach, wherein, in one round, I visit EVERY restaurant some, v times. I average out the rewards for each restaurant, and toss out half of the lowest performing ones. Now I repeat this strategy till I am left with one/few restaurants. This approach won't guarantee the best, but it will **probably** give me **approximately** the best. This is the **Median Elimination** approach.

