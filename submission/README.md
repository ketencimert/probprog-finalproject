# Glicko 2 Rating System

The goal of this project is to reproduce the Glicko 2 rating system using Stan. We plan to fit the model described in the original paper (Glickman, 2001) using different inference algorithms and different priors.

Glicko 2 considers the setting where there are various players with different merits/strength. The merits/strength of each player is latent. The aim of the model is to undercover this latent structure. That is, in Bayesian terms, making posterior inference over player merits/strength. This is particularly useful in settings where it is essential to evaluate the performance of the players.

## References

This is the final project by [Ozan Adiguzel](https://github.com/Ozan147) and [Mert Ketenci](https://github.com/ketencimert) for [Machine Learning with Probabilistic Programming](http://www.proditus.com/mlpp2021) (Fall 2021) at Columbia University.

  * [Glickman, M. E. (1999). Parameter estimation in large dynamic paired comparison experiments. Journal of the Royal Statistical Society: Series C (Applied Statistics), 48(3), 377-394.](http://www.glicko.net/research/glicko.pdf)
  * [Glickman, M. E. (2001). Dynamic paired comparison models with stochastic variances. Journal of Applied Statistics, 28(6), 673-689.](http://www.glicko.net/research/dpcmsv.pdf)
  * [Minka, T., Cleven, R., & Zaykov, Y. (2018). TrueSkill 2: An improved Bayesian skill rating system. Tech. Rep.](https://www.microsoft.com/en-us/research/uploads/prod/2018/03/trueskill2.pdf)
