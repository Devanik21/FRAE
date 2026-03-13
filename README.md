# FRAE-S: Fluid Resonance Advantage Estimation — Stable

### A Navier-Stokes Inspired Advantage Estimator for Actor-Critic Reinforcement Learning

**Devanik Debnath** · NIT Agartala, ECE  
[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![Status](https://img.shields.io/badge/Status-Experimental%20Research-orange?style=flat-square)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red?style=flat-square&logo=pytorch)]()

---

> **Note:** This is an experimental research prototype. No claims of state-of-the-art performance are made. The contribution is architectural and theoretical — a physics-motivated reformulation of advantage estimation in on-policy RL.

---

## Abstract

Standard advantage estimators in actor-critic methods — most notably Generalized Advantage Estimation (GAE) — use a fixed exponential decay parameter λ to trade off bias and variance. This parameter is a static scalar, insensitive to the local dynamics of the reward signal. FRAE-S proposes a **data-driven, physics-informed replacement**: the decay horizon at each timestep is governed by an *Information Reynolds Number* Re_t, computed from the momentum of TD-errors and a running estimate of their local variance. When Re_t falls below a critical threshold (analogous to laminar flow), credit propagates forward normally. When Re_t exceeds the threshold (turbulent flow), credit is **instantaneously truncated** — protecting the policy from accumulating gradient signal through high-variance, unstable transitions. The resulting advantage estimate is then normalized by the local variance estimate, providing an intrinsic stability sink without manual normalization.

---

## 1. Motivation

In standard GAE, the advantage at time $t$ is:

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD-error and $\lambda \in [0,1]$ is fixed throughout training.

**The problem:** $\lambda$ is a global hyperparameter blind to local signal quality. A single catastrophic TD-error spike contaminates the entire backward credit assignment through the full horizon. Meanwhile, in calm, low-variance regions, an unnecessarily truncated horizon wastes valid credit.

FRAE-S replaces this fixed scalar with a **per-timestep phase transition** driven by the local information dynamics.

---

## 2. The FRAE-S Formulation

### 2.1 TD-Error as Input Energy

$$\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$

where $d_t \in \{0,1\}$ is the episode termination flag.

### 2.2 Viscosity Invariant ζ_t

An exponential moving average tracks the local variance of TD-errors, playing the role of **fluid viscosity** — resistance to sudden state changes:

$$\sigma_t^2 = (1 - \alpha)\,\sigma_{t-1}^2 + \alpha\,\delta_t^2$$

$$\zeta_t = \sqrt{\sigma_t^2} + \varepsilon$$

where $\alpha \in (0,1)$ is the EMA decay rate and $\varepsilon$ is a small numerical stabilizer. $\zeta_t$ is always positive by construction.

### 2.3 Information Reynolds Number

The Reynolds number in classical fluid mechanics measures the ratio of inertial forces to viscous forces. Here, it measures the ratio of **TD-error momentum** (how abruptly the signal is changing) to **local stability** (how volatile the signal has been):

$$\text{Re}_t = \frac{|\delta_t - \delta_{t-1}|}{\zeta_t}$$

High Re_t indicates the TD-error is changing rapidly relative to its recent history — analogous to turbulent flow.

### 2.4 Phase Transition Switch β_t

A hard phase transition governed by a critical Reynolds number $\text{Re}_{\text{crit}}$:

$$\beta_t = \begin{cases} \beta_{\text{lam}} & \text{if } \text{Re}_t < \text{Re}_{\text{crit}} \quad \text{(Laminar: credit flows)} \\ 0 & \text{if } \text{Re}_t \geq \text{Re}_{\text{crit}} \quad \text{(Turbulent: truncation)} \end{cases}$$

where $\beta_{\text{lam}} \approx 0.95$ is the laminar propagation coefficient, analogous to $\lambda$ in GAE but only activated when the local flow is stable.

### 2.5 Fluid Advantage Energy U_t

The advantage is computed via a **backward pass** (causal temporal credit propagation):

$$U_t = \delta_t + \gamma \cdot \beta_t \cdot (1 - d_t) \cdot U_{t+1}, \qquad U_T = 0$$

This is structurally identical to GAE's backward recursion, except $\lambda$ is replaced by the data-driven $\beta_t$.

### 2.6 Normalized Advantage (Stability Sink)

$$\hat{A}_t = \frac{U_t}{\zeta_t}$$

Dividing by $\zeta_t$ performs **physical normalization**: in high-variance regions, large $U_t$ values are attenuated; in low-variance regions, small advantages are amplified. This eliminates the need for batch-level advantage normalization typically applied as a post-processing step.

---

## 3. Comparison to GAE

| Property | GAE | FRAE-S |
|----------|-----|--------|
| Horizon control | Fixed λ (scalar) | Data-driven β_t (per-step) |
| Normalization | Post-hoc batch norm | Intrinsic via ζ_t |
| Turbulence sensitivity | None | Re_t phase switch |
| Variance tracking | None | EMA viscosity σ_t² |
| Credit truncation | Soft (exponential decay) | Hard (instantaneous at Re_crit) |
| Hyperparameters | γ, λ | γ, α, Re_crit, β_lam |

FRAE-S strictly generalizes GAE: when $\text{Re}_t < \text{Re}_{\text{crit}}$ always (i.e., when the trajectory is uniformly low-variance), FRAE-S reduces to GAE with $\lambda = \beta_{\text{lam}}$ up to the $\zeta_t$ normalization.

---

## 4. Architecture

FRAE-S is implemented on top of a standard **decoupled A2C** backbone:

```
Observation s_t
      │
  ┌───┴────────────┐
  │   Shared Trunk  │   (Two separate MLPs)
  └──┬──────────┬──┘
     │          │
  Actor π_θ  Critic V_φ
  (logits)    (scalar)
     │          │
  Categorical  TD-error δ_t
  Distribution     │
     │          FRAE-S Engine
  Action a_t    β_t, ζ_t, Re_t
                    │
              Advantage Â_t
```

**Network:** 2-layer MLP with Tanh activations, hidden size 64, separate actor and critic heads.  
**Optimizer:** Adam, lr = 1e-3, gradient clipping at max_norm = 0.5.  
**Entropy bonus:** coefficient 0.01 for exploration regularization.

---

## 5. Loss Function

$$\mathcal{L} = \mathcal{L}_{\pi} + \mathcal{L}_{V} + \mathcal{L}_{H}$$

$$\mathcal{L}_{\pi} = -\mathbb{E}_t\left[\log \pi_\theta(a_t | s_t) \cdot \hat{A}_t\right]$$

$$\mathcal{L}_{V} = \frac{1}{2}\,\mathbb{E}_t\left[\left(R_t - V_\phi(s_t)\right)^2\right]$$

$$\mathcal{L}_{H} = -c_H \cdot \mathbb{E}_t\left[\mathcal{H}[\pi_\theta(\cdot | s_t)]\right]$$

where $R_t = \hat{A}_t + V_\phi(s_t)$ is the target return and $c_H = 0.01$.

---

## 6. Hyperparameters

| Symbol | Variable | Default | Role |
|--------|----------|---------|------|
| $\gamma$ | `gamma` | 0.99 | Discount factor |
| $\alpha$ | `alpha` | 0.01 | EMA rate for variance tracking |
| $\text{Re}_{\text{crit}}$ | `re_crit` | 2.0 | Phase transition threshold |
| $\beta_{\text{lam}}$ | `beta_laminar` | 0.95 | Laminar propagation coefficient |
| $\varepsilon$ | `eps` | 1e-8 | Numerical stabilizer |

---

## 7. Experiments

Preliminary experiments run on:

| Environment | Epochs | Steps/Epoch | Notes |
|-------------|--------|-------------|-------|
| `CartPole-v1` | 300 | 200 | Low-dimensional control baseline |
| `LunarLander-v3` | 1000 | 500 | High-variance, sparse reward |

Training telemetry exposes four physics-informed diagnostics per epoch: episodic reward, mean $\beta_t$ (laminar fraction), mean $\zeta_t$ (viscosity scale), and max $\text{Re}_t$ (peak turbulence). These provide interpretable insight into *why* the estimator truncates credit at specific points in training — something standard GAE offers no visibility into.

---

## 8. Telemetry Signals

Beyond reward curves, FRAE-S produces three diagnostically meaningful signals:

**Mean β_t** — the fraction of timesteps where credit flows forward. A declining β_t early in training indicates high turbulence (the value network is poor); a rising β_t over time signals the critic is converging and the flow is stabilizing.

**Mean ζ_t** — the average viscosity scale. Large ζ_t indicates a high-variance trajectory; as the policy improves and returns become more predictable, ζ_t shrinks.

**Max Re_t** — the peak information Reynolds number in the epoch. Persistent high values indicate the policy is encountering transitions it cannot yet predict. Correlation with reward drops can diagnose policy collapse early.

---

## 9. Limitations & Open Questions

- The phase transition is a **hard switch** at Re_crit. A soft sigmoid transition may reduce sensitivity to the threshold hyperparameter.
- The EMA variance $\sigma_t^2$ is computed **forward-in-time**, meaning early timesteps in a rollout have less accurate variance estimates. A two-pass algorithm could address this.
- Re_crit = 2.0 is set heuristically. Whether a theoretically justified choice exists (analogous to Re ≈ 4000 in pipe flow) is an open question.
- No formal sample complexity or convergence analysis has been conducted. FRAE-S is a **research prototype**, not a production algorithm.
- Comparison against PPO + GAE under controlled hyperparameter tuning is pending.

---

## 10. Running the Code

```bash
pip install torch gymnasium matplotlib numpy
pip install gymnasium[box2d]      # for LunarLander

python liquid_ns.py
```

The script sequentially trains on CartPole-v1 (300 epochs) and LunarLander-v3 (1000 epochs), then renders the physics telemetry plots.

---

## 11. Citation

If you build on this work, please cite informally as:

```
Devanik Debnath. FRAE-S: Fluid Resonance Advantage Estimation — Stable.
Experimental Research Prototype, NIT Agartala, 2025–2026.
GitHub: github.com/Devanik21
```

Formal arXiv submission and ablation study pending.

---

## References

- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016.
- Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*, ICML 2016.
- Reynolds, O., *An Experimental Investigation of the Circumstances Which Determine Whether the Motion of Water Shall Be Direct or Sinuous*, Phil. Trans. R. Soc. 1883.
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press 2018.

---

*Experimental. Unreviewed. All physics analogies are structural motivations, not physical laws.*  
*— Devanik Debnath, NIT Agartala*
