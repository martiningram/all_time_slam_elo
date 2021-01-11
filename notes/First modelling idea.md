

## First modelling idea

The general framework for this model is something called a Bradley-Terry model. The idea is that each player has a _fixed_ skill, call it $\theta_i$ for player $i$, say. When players $i$ and $j$ meet, the probability that $i$ wins is given by:
$$
p(\textrm{i wins} | \theta_i, \theta_j) = \frac{1}{1 + e^{-(\theta_i - \theta_j)}} = \textrm{logit}^{-1}(\theta_i - \theta_j)
$$
This is actually essentially the same as Elo's win probability, except rescaled and shifted. The 1500 value is at zero. Also, a difference of 1 on the logit scale here is about equal to a difference of 175 Elo points.

A simple model might assume that the _prior_ for each player is given by
$$
\theta_i \sim \mathcal{N}(0, \sigma^2).
$$
This would say that, without knowing anything about a new player, we would guess their rating to be zero, with some uncertainty around that given by $\sigma$. So if $\sigma$ is 1, for example, we'd expect a new player's rating to be between $-2$ and $2$ with about 95% probability. Note that this model is _static_: we're not modelling any change in ratings over time like Elo. Ultimately it would be nice to model dynamic skills, but given the large number of players, this seems like a good start.

That model works OK, but it doesn't really do what we want. In particular, it doesn't tell us anything about varying depth and skill over time. We'd treat a player playing their first match at Wimbledon in 1890 the same as a player playing their first in 1970. We probably expect that the mean has shifted, and also that the uncertainty might have changed. For example, if there was a large range of skills in 1890 because a bunch of random people entered in addition to good players, we'd expect a big range of skills, while these days, we might expect a smaller range since only the best make it.

A first modelling idea is the following:
$$
\theta_i \sim \mathcal{N}(\mu_i, \sigma^2_i).
$$
We model $\mu_i$ as follows:
$$
\mu_i = \mu_{t(i), y(i)} + \beta_\mu (\log n_{t(i), y(i)} - \log \bar{n})
$$
And the standard deviation $\sigma_i$ as follows:
$$
\log \sigma_i = \alpha_{t(i), y(i)} + \beta_\sigma (\log n_{t(i), y(i)} - \log \bar{n})
$$
What this means is that player $i$ is given prior mean $\mu_{t(i), y(i)}$, a prior mean specific to their first tournament, $t(i)$, and year on tour $y(i)$. The same holds for the prior uncertainty $\sigma^2$. This allows us to model different prior means and variances for different years and tournaments. 

The second terms involving $\beta$ model the influence of the draw size: $n_{t(i), y(i)}$ is the total number of matches for a tournament that year, and $\log \bar{n}$ is the mean log number of matches per tournament across the dataset. The idea here is that if the tournament had a big draw, we might expect that the overall prior mean is going to be lower (more players --> less selective) and the overall prior variance might be larger (more players --> bigger range of skills), and the $\beta$ coefficients model this. I added this on Steph's suggestion and I think it's important.

The last component of this first model encodes the idea that we don't expect these means and variances to change very rapidly. What I mean is that the prior belief for a player playing Wimbledon in 1890 should probably be fairly close to that in 1891. We can encode this with a _random walk prior_:
$$
\mu_{t, y} \sim \mathcal{N}(\mu_{t - 1, y - 1}, \sigma^2_{mean})
$$
This says that the mean in the next year is a small random jump from that in the previous year. A similar random walk is on the variances.

The actual model is a little more complicated since I'm currently doing a _multivariate_ random walk on the tournaments, which means that the changes from year to year between the tournaments are correlated. I don't think this is really necessary, but it was the easiest thing to code in pymc3. The full model code is here:

```python
with pm.Model() as hierarchical_model:
    # Hyperprior on the random walk covariance for the prior means  
    mean_chol, _, _ = pm.LKJCholeskyCov(
        "mean_chol", n=n_tourneys, eta=2.0, sd_dist=pm.HalfNormal.dist(1.0), compute_corr=True
    )
    
    # Random walk on prior means
    prior_means = pm.MvGaussianRandomWalk('prior_means', chol=mean_chol, shape=(n_years-1, n_tourneys))
    
    # In first year, prior means are all zero
    full_prior_means = pm.math.concatenate([np.zeros((1, n_tourneys)), prior_means], axis=0)
    
    # Factor in draw size
    draw_size_mean_factor = pm.Normal('mean_draw_size_factor', mu=0., sigma=1.)
    full_prior_means = full_prior_means + draw_size_mean_factor * (log_draw_sizes.T - mean_log_draw_size)
    
    # Initial standard deviation of skills
    init_sd = pm.HalfNormal('init_sd', 1., shape=(1, n_tourneys))
    
    # Hyperprior on random walk covariance for prior (log) standard deviations
    sd_chol, _, _ = pm.LKJCholeskyCov(
        "sd_chol", n=n_tourneys, eta=2.0, sd_dist=pm.HalfNormal.dist(1.0), compute_corr=True
    )
    
    # Random walk on prior variances
    prior_log_sds = pm.MvGaussianRandomWalk('prior_log_sds', chol=sd_chol, shape=(n_years-1, n_tourneys))
    full_prior_log_sds = pm.math.concatenate([pm.math.log(init_sd), prior_log_sds], axis=0)
    
    # Factor in draw size
    full_prior_log_sds = full_prior_log_sds + draw_size_sd_factor * (log_draw_sizes.T - mean_log_draw_size)
    full_prior_sds = pm.math.exp(full_prior_log_sds)
        
    # Prior
    player_skills = pm.Normal('player_skills', 
                              mu=full_prior_means[first_seen_ids, first_seen_tourney_ids], 
                              sigma=full_prior_sds[first_seen_ids, first_seen_tourney_ids], shape=n_players)

    # Likelihood
    logit_skills = player_skills[winner_ids] - player_skills[loser_ids]
    
    lik = pm.Bernoulli('win_lik', logit_p=logit_skills, observed=np.ones(winner_ids.shape[0]))
```

### Results

Quick caveat: not all the $\hat{R}$ statistics look good here (particularly those for the prior mean), so some of these values may not have converged, so it's probably best not to over-interpret things. Here's the maximum $\hat{R}$ for each set of parameters:

```
mean_chol_cholesky-cov-packed__    1.04
prior_means                        2.14
mean_draw_size_factor              1.05
init_sd_log__                      1.13
sd_draw_size_factor                1.07
sd_chol_cholesky-cov-packed__      1.10
prior_log_sds                      1.08
player_skills                      1.99
```

So it looks like everything apart from the prior means and player skills is roughly OK ($\hat{R}$ should be below 1.1), but those two aren't really. We'll have to look into that.

![prior_mean_with_draw_size](/home/martin/projects/all_time_elo/jupyter/prior_mean_with_draw_size.png)

The prior mean seems to be generally rising over time, which is maybe intuitive: each year improves a little bit on the previous one, generally speaking.

![prior_sd_with_draw_size](/home/martin/projects/all_time_elo/jupyter/prior_sd_with_draw_size.png)

The prior standard deviation is pretty interesting, starting out large (i.e. a large range of skills) and then dropping dramatically until about 1980, since when it's been fairly flat. I find the steep decline of the AO here pretty intriguing, from about 1951 onwards. I wonder if that's when travel started to become more affordable, or something?

The draw size coefficients end up being $\beta_\mu = -0.4$ and $\beta_\sigma = 0.09$. So that makes sense I think: bigger draw, lower prior mean and larger range of skills.

What about the player skills? Again, given the lack of convergence, probably shouldn't look too much. But here are the top 10 highest skills:

|                         | mean  | sd    |
| ----------------------- | ----- | ----- |
| Don Budge               | 5.178 | 0.644 |
| Bobby Riggs             | 4.882 | 0.691 |
| Bill Tilden Ii          | 4.877 | 0.542 |
| Frederick Ted Schroeder | 4.839 | 0.715 |
| Jack Kramer             | 4.769 | 0.659 |
| Jean Rene Lacoste       | 4.719 | 0.553 |
| Joe Hunt                | 4.655 | 0.729 |
| Bill Johnston           | 4.590 | 0.582 |
| Henri Jean Cochet       | 4.537 | 0.543 |
| Novak Djokovic          | 4.517 | 0.526 |

So some of these old players do extremely well! _But_, I was curious how things would change if instead, we compute a z-score. The idea here is that back in the 20s, where the range of skills was really high, someone with a rating of 4 would be much less surprising than now, where the range of skills is much smaller. Here, I compute the z-score by subtracting the player's prior mean (a function of which year and tournament they played, as discussed), and dividing by their prior standard deviation. The result should tell us something about how much better a player is than would have been expected from when and where they started.

The result is then:

```
Bill Tilden Ii                 3.50
Roger Federer                  3.45
Bjorn Borg                     3.44
Rafael Nadal                   3.34
Bill Johnston                  3.19
Fred Perry                     3.13
Novak Djokovic                 3.12
Jean Rene Lacoste              3.11
Maurice Evans Mcloughlin       3.09
Henri Jean Cochet              2.99
Ivan Lendl                     2.96
Jaroslav Drobny                2.83
Pete Sampras                   2.81
John McEnroe                   2.78
Andy Murray                    2.76
Jimmy Connors                  2.72
Lestocq Robert Erskine         2.71
Arthur Ashe                    2.68
Don Budge                      2.67
Jean Laurent Robert Borotra    2.66
```

I think that's sort of a fun list. But this is very experimental and only a first idea.