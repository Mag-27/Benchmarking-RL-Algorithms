# 🤖 Benchmarking RL Algorithms

## Value-Based and Policy-Gradient/Actor-Critic Deep RL, From Scratch

![Language](https://img.shields.io/badge/Language-Python%203.11%2B-3776AB?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square)
![Environments](https://img.shields.io/badge/Environments-Gymnasium%20%2B%20MuJoCo-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Results%20In%20Progress-yellow?style=flat-square)

**From-scratch PyTorch implementations of 14 deep RL algorithms, spanning value-based and policy-gradient/actor-critic families, benchmarked on classic-control and MuJoCo continuous-control tasks.**

Everything — networks, replay buffers, exploration schedules, multiprocess rollout workers — is implemented directly on top of `torch`, with no RL library (Stable-Baselines3, RLlib, etc.) in the loop. Two notebooks: one for value-based methods, one for policy-gradient and actor-critic methods.

## Algorithms

### `value_based_deep_rl.ipynb`

All five agents are trained and evaluated on both environments below.

| Algorithm | Environment(s) | Original paper |
|---|---|---|
| NFQ (Neural Fitted Q-Iteration) | CartPole-v0, MountainCar-v0 | Riedmiller, "Neural Fitted Q Iteration – First Experiences with a Data Efficient Neural Reinforcement Learning Method" (ECML 2005) — no arXiv preprint |
| DQN | CartPole-v0, MountainCar-v0 | Mnih et al., ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602) (arXiv:1312.5602) |
| Double DQN (DDQN) | CartPole-v0, MountainCar-v0 | van Hasselt et al., ["Deep Reinforcement Learning with Double Q-learning"](https://arxiv.org/abs/1509.06461) (arXiv:1509.06461) |
| Dueling Double DQN (D3QN) | CartPole-v0, MountainCar-v0 | Wang et al., ["Dueling Network Architectures for Deep Reinforcement Learning"](https://arxiv.org/abs/1511.06581) (arXiv:1511.06581) |
| D3QN + Prioritized Experience Replay (D3QN-PER) | CartPole-v0, MountainCar-v0 | Schaul et al., ["Prioritized Experience Replay"](https://arxiv.org/abs/1511.05952) (arXiv:1511.05952), applied on top of the D3QN architecture above |

### `policy_gradient_actor_critic.ipynb`

| Algorithm | Environment | Original paper |
|---|---|---|
| REINFORCE | Pendulum-v1 | Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (Machine Learning, 1992) — no arXiv preprint |
| VPG (Vanilla Policy Gradient) | Pendulum-v1 | Sutton et al., "Policy Gradient Methods for Reinforcement Learning with Function Approximation" (NeurIPS 1999/2000) — no arXiv preprint |
| A2C | Hopper-v4 | No standalone paper — synchronous variant of A3C (below), popularized by OpenAI Baselines |
| A3C | Hopper-v4 | Mnih et al., ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/abs/1602.01783) (arXiv:1602.01783) |
| GAE (Generalized Advantage Estimation) | Hopper-v4 | Schulman et al., ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438) (arXiv:1506.02438) — implemented as `compute_gae()` and used inside the A2C training loop, not a standalone agent |
| DDPG | HalfCheetah-v4 | Lillicrap et al., ["Continuous Control with Deep Reinforcement Learning"](https://arxiv.org/abs/1509.02971) (arXiv:1509.02971) |
| TD3 | HalfCheetah-v4 | Fujimoto et al., ["Addressing Function Approximation Error in Actor-Critic Methods"](https://arxiv.org/abs/1802.09477) (arXiv:1802.09477) |
| SAC | HalfCheetah-v4 | Haarnoja et al., ["Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"](https://arxiv.org/abs/1801.01290) (arXiv:1801.01290) |
| PPO | HalfCheetah-v4 | Schulman et al., ["Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347) (arXiv:1707.06347) |

## Setup

Both notebooks import: `gymnasium`, `numpy`, `matplotlib`, `torch` (`nn`, `nn.functional`, `optim`), plus standard library modules (`collections`, `random`, `time`, `itertools`). `value_based_deep_rl.ipynb` additionally imports `pandas`. `policy_gradient_actor_critic.ipynb` additionally imports `torch.multiprocessing` (for A3C/A2C/PPO) and `torch.distributions` (`Normal`, `Categorical`).

Both notebooks install environments via:
```
!pip install gymnasium
!pip install swig
!pip install gymnasium[box2d]
!pip install gymnasium[mujoco]
```
No version pins are given in these install cells. MuJoCo (`gymnasium[mujoco]`) is required for the Hopper-v4 and HalfCheetah-v4 runs in `policy_gradient_actor_critic.ipynb`.

Interpreter/library versions, read from cell outputs:
- `value_based_deep_rl.ipynb` was executed under Python 3.11 (`C:\Python311\...` appears in a warning traceback).
- `policy_gradient_actor_critic.ipynb` shows execution under at least two different environments across its cells: a Windows Conda environment named `torchtest` (`anaconda3\envs\torchtest\...`) for most sections, and Python 3.12 (`/usr/local/lib/python3.12/dist-packages/...`, a Colab-style path) for the A2C/A3C/GAE sections. The notebook is not tied to a single Python version.
- <!-- TODO: version --> exact `torch`, `gymnasium`, and `numpy`/`pandas` versions — no `__version__` prints or `pip freeze`/`pip show` output exist in either notebook.

## How to run

Open whichever notebook matches what you want to reproduce — `value_based_deep_rl.ipynb` for the classic-control value-based agents, `policy_gradient_actor_critic.ipynb` for the continuous-control policy-gradient/actor-critic agents. Each notebook is self-contained (imports → environment setup → hyperparameters → helpers → agent classes → experiments) and should be run top to bottom.

Hyperparameters:
- In `value_based_deep_rl.ipynb`, all shared hyperparameters (`gamma`, `epsilon`, `temp`, `delta`, `tau`, `alpha`, `beta`, `beta_rate`, `MAX_TRAIN_EPISODES`, `MAX_EVAL_EPISODES`) are centralized in one cell in the "Hyperparameters" section near the top, ahead of any agent code, so they're easy to sweep.
- In `policy_gradient_actor_critic.ipynb`, the top-level "Hyperparameters" cell is present but left as a commented-out stub; the actual hyperparameters are centralized per algorithm instead, each in its own clearly-labeled cell near that algorithm's section (e.g. `a3c_*`, `a2c_*`, `ppo_*` variables, and inline keyword arguments for REINFORCE/VPG/DDPG/TD3/SAC).

Agent classes are built up incrementally: the same class name is re-opened across several consecutive cells (e.g. `class DQN():` then `class DQN(DQN):` then `class DQN(DQN):` again), with each cell adding another method, so the full class is only complete once all of its cells have run in order.

Seeds:
- `value_based_deep_rl.ipynb`: every agent/environment pair (NFQ, DQN, DDQN, D3QN, D3QN-PER × CartPole-v0, MountainCar-v0) is run over **5 seeds**; plots show the mean with a min/max envelope across seeds.
- `policy_gradient_actor_critic.ipynb`: seed counts vary by algorithm — REINFORCE and VPG use a fixed list of **5 seeds** (`SEEDS = [42, 123, 7, 2024, 99]`); DDPG, TD3, and PPO each loop over **5 seeds** (`range(5)`); SAC is run over **3 runs** (`num_runs = 3`); the individual A2C and A3C Hopper runs, the GAE λ-sweep, and the A2C-vs-A3C comparison are each a **single run** (no seed loop).

## Implementation notes

- **Shared, parameterized replay buffer** (value-based notebook): a single `ReplayBuffer` class (built up across cells, per the pattern above) is reused by NFQ, DQN, DDQN, D3QN, and D3QN-PER via a `bufferType` argument (`'DQN'`, `'DDQN'`, `'D3QN'`, `'PER-D3QN'`), rather than each agent having its own buffer implementation.
- **Prioritized replay with importance-sampling weights**: D3QN-PER's buffer stores per-transition priorities, samples proportionally to priority, and computes IS weights (`weights = (1/N · 1/P(i))^β`, normalized by `max(weights)`) that are multiplied into the TD-error loss (`loss = mean(weights * loss_fn(qs, tdTargets))`), with `alpha`, `beta`, and `beta_rate` as tunable hyperparameters.
- **Dueling architecture**: `createDuelingNetwork` splits into value and advantage streams and recombines them with the advantage mean subtracted out (`Q = V + (A − mean(A))`), used by D3QN and D3QN-PER.
- **Multiprocess workers with shared-memory rollout buffers**: A3C, A2C, and PPO all use `torch.multiprocessing`, with the global network moved to shared memory (`.share_memory()`) and per-step rollout tensors (observations, actions, log-probs, rewards, dones, values) pre-allocated with `.share_memory_()` so worker processes write directly into buffers the main process reads from, coordinated via `mp.Queue()`. A3C additionally implements a custom `SharedAdam` optimizer that puts Adam's per-parameter moment buffers in shared memory so all workers update the same momentum state.
- **GAE**: implemented as a standalone `compute_gae()` function (bias/variance controlled by `gae_lambda`) and wired into the A2C training loop via a `use_gae` flag; a separate cell sweeps `gae_lambda ∈ {0.0, 0.5, 0.95, 1.0}` on Hopper-v4 and plots the resulting advantage estimates and their variance/mean-absolute-value trade-off.
- **Twin critics and delayed policy updates**: TD3 maintains two online/target critic pairs (`onlineValueNetwork1/2`, `targetValueNetwork1/2`) and updates the policy only every `trainPolicyFrequency` steps; SAC also takes the min over two online critics (`torch.min(q_current_1, q_current_2)`) when computing the policy loss.
- **Entropy tuning**: SAC learns its entropy coefficient rather than fixing it — `logAlpha` is a parameter of the policy network with its own optimizer (`alphaOptimizerFn`), and `alpha = exp(logAlpha)` is used in both the policy loss and the target Q computation. A2C, A3C, and PPO instead use a fixed entropy-bonus coefficient (`*_ent_coef`) added to the loss.

## Results

<!-- TODO: fill in results -->

Plot outputs actually present in the committed notebooks:

**`value_based_deep_rl.ipynb`**: the notebook's execution stops before its final "Experiments and Plots" comparison section — cells that call `plotQuantity(...)` to compare NFQ/DQN/DDQN/D3QN/D3QN-PER across both environments (the 9 plots described in that section's markdown) show no cell outputs. Only one plot image exists anywhere in the notebook (in the DQN/MountainCar section), and it is emitted immediately after a `ValueError: x and y must have same first dimension, but have shapes (1000,) and (18000,)` from the preceding `plotQuantity(...)` call in the same cell — the figure it belongs to is unclear. The three "Observations" markdown cells at the end of the notebook (CartPole comparison, MountainCar comparison, general findings) are all unfilled `> TODO: write up observations from the plots above.` placeholders.

**`policy_gradient_actor_critic.ipynb`**: seven plot images are present:
1. "A3C Training on Hopper-v4" — two panels, episode rewards and cumulative mean reward.
2. "A2C Training — Hopper-v4" — two panels, episode returns and cumulative mean return.
3. "GAE Advantage Estimates for Different λ Values" — advantage estimate vs. timestep, one line per λ.
4. "Bias–Variance Trade-off vs λ" — variance and mean-absolute-advantage vs. λ, on synthetic (non-training) data.
5. "GAE Lambda Sweep — Hopper-v4" — two panels (episode returns, cumulative mean return) comparing A2C trained with λ ∈ {0.0, 0.5, 0.95, 1.0}.
6. "A2C vs A3C — Hopper-v4" — two panels comparing A2C (GAE λ=0.95) against A3C.
7. "SAC — HalfCheetah-v4" — two panels, train returns (mean ± std band over 3 runs) and cumulative mean return.

Two comparison sections do not have plot output in the committed notebook: the REINFORCE-vs-VPG comparison on Pendulum-v1 (the run cells raise `NameError: name 'REINFORCE' is not defined` / `NameError: name 'VPG' is not defined`, so `plot_pendulum(...)` is never reached), and the final DDPG-vs-TD3-vs-PPO comparison on HalfCheetah-v4 (the plotting cell has no captured output). That final cell does contain an author-written `obs_text` panel (visible in source, not confirmed rendered) with hand-written claims about TD3 vs. DDPG stability and PPO vs. off-policy sample efficiency — treat these as unverified notes rather than results read off a rendered plot.
