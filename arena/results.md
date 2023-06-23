# Preset Arena results

Description of the experiment: [Preset Arena: 17,205 comparisons between 241 different presets.](https://www.reddit.com/r/LocalLLaMA/comments/14adfw2/preset_arena_17205_comparisons_between_241/)

Some numbers:

* 6733 valid votes
* 934 voting sessions
* 203 users with usernames

## Final results

* Sorted by instruct performance: [instruct](https://oobabooga.github.io/arena/instruct.html)
* Sorted by chat performance: [chat](https://oobabooga.github.io/arena/chat.html)

## Dataset

Soon

## Ranking the presets

### Bad voting sessions

The first step in the analysis of the votes was to try to identify suspicious voters. Each voting session received a unique uuid string, allowing the frequency of left/right votes to be analyzed.

I have used the following code to calculate the probability that a voting session was biased. It was obtained by asking ChatGPT for a fair coin test:

```python
from scipy.stats import beta

def compute_bias_probability(outcomes, prior_alpha=1, prior_beta=1, _print=False):
    # Count the number of heads and tails
    num_heads = outcomes.count('left')
    num_tails = outcomes.count('right')

    if _print:
        print(num_heads, num_tails)

    # Update the prior with the observed outcomes
    posterior_alpha = prior_alpha + num_heads
    posterior_beta = prior_beta + num_tails

    # Calculate the bias probability using the Beta distribution
    bias_probability = beta.cdf(0.5, posterior_alpha, posterior_beta)

    return bias_probability
```

A session was disconsidered if `bias_probability > 0.95`, which happened for 2.4% of all sessions.

### Estimating the elo scores

The basic formula is

```python
def update_rating(rating, opponent_rating, outcome, k=32):
    expected_score = 1 / (1 + 10**((opponent_rating - rating) / 400))
    new_rating = rating + k * (outcome - expected_score)
    return new_rating
```

where the ratings are initialized as `1000` for all presets, and `outcome` is 1 in case of winning and 0 in case of losing.

To make things more robust, I have used the following procedure instead of calculating the elo scores just once:

* take a random subsample containing 90% of votes
* using that sample, calculate the elo scores for chat and instruct prompts separately
* repeat 100 times
* take the averages of the elo scores for each preset

### Comments

1) I find that the chat presets are all kind of the same. It may be due to the chat prompts being too simple and short, causing presets with low top_p to be favored.

2) 5 variations of the Mirostat preset were included. It turned out that `Mirostat-5` was better than the `Mirostat` preset originally included in text-generation-webui:

<table><tr><th>preset</th><th>params</th><th>elo score (chat)</th><th>elo score (instruct)</th><th>elo score (all)</th><th>matches (chat)</th><th>matches (instruct)</th></tr><tr><td>Mirostat-5</td><td>2</td><td>1026.5014937016579</td><td>1095.5957306337566</td><td>1061.0486121677072</td><td>30</td><td>22</td></tr><tr><td>Mirostat</td><td>1</td><td>987.4284290976724</td><td>1091.528754921891</td><td>1039.4785920097818</td><td>25</td><td>20</td></tr><tr><td>Mirostat-2</td><td>2</td><td>1058.6173333941947</td><td>1030.0263508768217</td><td>1044.3218421355082</td><td>27</td><td>25</td></tr><tr><td>Mirostat-3</td><td>2</td><td>979.0888416948233</td><td>1014.594027476248</td><td>996.8414345855356</td><td>28</td><td>30</td></tr><tr><td>Mirostat-4</td><td>2</td><td>1023.2718233979415</td><td>1009.3037137304021</td><td>1016.2877685641718</td><td>31</td><td>31</td></tr></table>

3) Similarly, 5 Contrastive Search variations were included, `Contrastive Search-3` ended up being a bit better than the original `Contrastive Search`:

<table><tr><th>preset</th><th>params</th><th>elo score (chat)</th><th>elo score (instruct)</th><th>elo score (all)</th><th>matches (chat)</th><th>matches (instruct)</th></tr><tr><td>Special-Contrastive Search-3</td><td>3</td><td>1081.6638160617522</td><td>1109.6104645202452</td><td>1095.6371402909986</td><td>26</td><td>19</td></tr><tr><td>Special-Contrastive Search</td><td>3</td><td>1082.8334196036967</td><td>1084.7923688591343</td><td>1083.8128942314156</td><td>29</td><td>29</td></tr><tr><td>Special-Contrastive Search-1</td><td>3</td><td>926.8246947947172</td><td>859.149252796177</td><td>892.986973795447</td><td>13</td><td>11</td></tr><tr><td>Special-Contrastive Search-4</td><td>3</td><td>766.2481498240295</td><td>792.4678047191803</td><td>779.3579772716049</td><td>31</td><td>18</td></tr><tr><td>Special-Contrastive Search-2</td><td>3</td><td>819.1460361418351</td><td>747.206551560756</td><td>783.1762938512956</td><td>21</td><td>24</td></tr></table>

4) Eta Sampling (another special technique), by itself, did not perform very well:

<table><tr><th>preset</th><th>params</th><th>elo score (chat)</th><th>elo score (instruct)</th><th>elo score (all)</th><th>matches (chat)</th><th>matches (instruct)</th></tr><tr><td>Special-Eta Sampling</td><td>3</td><td>1016.2128130022467</td><td>1014.1492450072842</td><td>1015.1810290047654</td><td>28</td><td>23</td></tr></table>

5) The best overall preset, considering the average of the chat and instruct elo scores, was also perhaps the most obvious. I originally named it `simple-1` not expecting it to get anywhere:

```
temperature: 0.7
top_p: 0.9
repetition_penalty: 1.15
top_k: 20
```

The StarChat preset, also very simple, also performed well:

```
temperature: 0.2
top_p: 0.95
top_k: 50
```

This demonstrates that fancy samplers may not be that necessary.

### Presets that I chose

For the purpose of including better presets in text-generation-webui, I removed presets with `top_p < 0.05` or `top_k < 3` because that seemed too low and artificial. That left me with the following (in decreasing order of elo score):

#### Instruct

| Preset | New name |
|------|---------|
| simple-1 | |
| random_preset_066 | Divine Intellect
| starchat | |
| random_preset_035 | Space Alien
| random_preset_002 | Asterism
| random_preset_183 | Titanic
| Special-Contrastive Search-3 | |
| random_preset_027 | Shortwave |
| random_preset_134 | Big O |
| tfs-with-top-a |

#### Chat

| Preset | New name |
|------|---------|
| random_preset_101 | Midnight Enigma |
| random_preset_161 | Yara |
| Kobold-Godlike | |

I took the liberty of giving gave some cheesy names for the new random presets.

### Sampler frequency

In those 13 new presets, these are the sampling parameters that are present and how many times they appear:

```
     12 temperature
     11 top_p
     11 top_k
     11 repetition_penalty
      5 top_a
      3 typical_p
      3 tfs
      3 eta_cutoff
      1 penalty_alpha
      1 epsilon_cutoff
      1 encoder_repetition_penalty
      1 do_sample
```
