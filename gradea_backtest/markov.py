"""Markov-switching volatility regimes.

Ported from the `MarkovRegime` class in Roman Paolucci's (QuantGuild)
"Markov Chain Regime Switching Bot" lectures, parts 1 and 2. The algorithm is
his; three adaptations were needed to make it safe for a backtest, and they are
documented below because two of them change what the output means.

**Why this belongs here.** Every other regime family in this package is a rule
someone wrote down: a curve is inverted when the slope is negative, conditions
are tight when NFCI is positive. Those thresholds are defensible but they are
assumptions, not findings. This classifier instead *learns* where the regime
boundaries sit from the data, and carries a probability rather than a hard
label, so a day that sits between regimes is reported as sitting between them.

**The mechanism.** Three hidden states emit an observed volatility through
state-specific Gaussians. A transition matrix with a heavy diagonal makes states
sticky, which is the empirical fact the model exists to exploit: calm periods
cluster, and so do turbulent ones. Belief is carried forward by Bayes' rule --
predict through the transition matrix, weight by the likelihood of what was
actually observed, renormalise.

Adaptations
-----------
1. **The observation.** The original reads intraday range, ``(high - low) /
   close``, from live Interactive Brokers bars. The archive has one number per
   day, so the daily analogue used here is the absolute change in the 10-year
   yield. It measures the same thing -- how far the market travelled -- at the
   only resolution available.

2. **Recalibration is expanding, not once.** This is the load-bearing change.
   The original calibrates on a block of history and then runs live, which is
   correct for a live bot: everything it was fitted on really is in its past.
   Fitting once over a *backtest* sample is a different thing entirely, because
   the emission means and transition matrix would then encode the 2008 and 2020
   volatility spikes into the model's 1985 labels. Here the model is refitted
   every ``recalibrate_every`` days using only observations up to that point, and
   the filter runs forward between refits. Labels are therefore reproducible
   from data that existed when they were assigned, which
   ``tests/test_no_lookahead.py`` checks by deleting the future and requiring
   identical output.

3. **Filtering only, never smoothing.** The forward filter is kept exactly as
   written, and deliberately: the obvious "improvement" is to run
   Baum-Welch or Viterbi over the whole series to get better labels, and that
   would be a lookahead of the worst kind -- the labels would look superb and
   every strategy conditioned on them would be untradeable. Filtering answers
   "what regime am I in, given what I have seen", which is the only question a
   trader can ask.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Sticky by construction: a regime is far more likely to persist than to
#: switch, and a jump from calm straight to turbulent is rare. These are the
#: original's priors, used until enough history accrues to estimate them.
DEFAULT_TRANSITION = np.array(
    [
        [0.90, 0.08, 0.02],  # from Low
        [0.10, 0.80, 0.10],  # from Medium
        [0.02, 0.08, 0.90],  # from High
    ]
)

STATE_LABELS = ("Low vol", "Medium vol", "High vol")

#: Minimum observations before the model is fitted at all. Below this the
#: emission parameters are estimated from too few points to mean anything.
MIN_CALIBRATION = 250


@dataclass
class MarkovVolatilityRegime:
    """Three-state Markov-switching classifier over a volatility observation.

    Attributes
    ----------
    n_states:
        Fixed at three. The original's choice, kept: two states cannot express
        "neither calm nor stressed", and four cannot be estimated reliably from
        the number of genuine volatility cycles in any realistic sample.
    state_probs:
        The belief vector, ``P(state = i | observations so far)``. Sums to one.
    """

    n_states: int = 3
    transition_matrix: np.ndarray = field(default_factory=lambda: DEFAULT_TRANSITION.copy())
    emission_means: np.ndarray = field(default_factory=lambda: np.array([0.0005, 0.002, 0.005]))
    emission_stds: np.ndarray = field(default_factory=lambda: np.array([0.0003, 0.001, 0.003]))
    state_probs: np.ndarray = field(default_factory=lambda: np.full(3, 1 / 3))

    def calibrate(self, observations: np.ndarray) -> bool:
        """Fit emissions and transitions to a window of past observations.

        Returns False and leaves the model untouched when the window is too
        short to estimate anything -- a silently half-fitted model is worse than
        an unfitted one.
        """
        obs = np.asarray(observations, dtype="float64")
        obs = obs[np.isfinite(obs) & (obs > 0)]
        if obs.size < MIN_CALIBRATION:
            return False

        # Seed the state assignment by splitting the observed distribution into
        # thirds. Because the split is by percentile, the resulting emission
        # means are monotone in the state index by construction, so Low really
        # is the low-volatility state and the transition matrix estimated below
        # is indexed consistently with them.
        p33, p67 = np.percentile(obs, [33, 67])
        assignment = np.zeros(obs.size, dtype=int)
        assignment[obs >= p33] = 1
        assignment[obs >= p67] = 2

        means = self.emission_means.copy()
        stds = self.emission_stds.copy()
        for state in range(self.n_states):
            block = obs[assignment == state]
            if block.size >= 3:
                means[state] = float(block.mean())
                # A zero standard deviation would make the likelihood a spike
                # and the filter would lock onto one state permanently.
                stds[state] = float(max(block.std(), 1e-9))

        # Transitions counted off the seed assignment, with Laplace smoothing so
        # a transition never seen in the window is improbable rather than
        # impossible. Without it one unlucky window makes a state absorbing.
        counts = np.zeros((self.n_states, self.n_states))
        for t in range(1, assignment.size):
            counts[assignment[t - 1], assignment[t]] += 1
        matrix = self.transition_matrix.copy()
        for i in range(self.n_states):
            row = counts[i].sum()
            if row > 0:
                matrix[i] = (counts[i] + 0.1) / (row + 0.3)

        self.emission_means, self.emission_stds, self.transition_matrix = means, stds, matrix
        return True

    def _likelihoods(self, value: float) -> np.ndarray:
        """Gaussian emission density of ``value`` under each state."""
        z = (value - self.emission_means) / self.emission_stds
        return np.exp(-0.5 * z * z) / (self.emission_stds * np.sqrt(2 * np.pi))

    def step(self, value: float) -> np.ndarray:
        """Advance the belief one observation and return the new belief vector.

        Predict through the transition matrix, weight by how likely each state
        makes what was actually observed, renormalise. A non-finite observation
        advances the prediction without conditioning on anything, which is the
        correct handling of a missing day rather than a reason to guess.
        """
        predicted = self.state_probs @ self.transition_matrix
        if not np.isfinite(value) or value <= 0:
            self.state_probs = predicted / predicted.sum()
            return self.state_probs

        posterior = predicted * self._likelihoods(value)
        total = posterior.sum()
        if not np.isfinite(total) or total <= 0:
            # Every state calls the observation impossible -- far outside all
            # three Gaussians. Fall back to the prediction rather than dividing
            # by zero; the next recalibration will widen the emissions.
            self.state_probs = predicted / predicted.sum()
        else:
            self.state_probs = posterior / total
        return self.state_probs


def volatility_observation(pit: pd.DataFrame, column: str = "DGS10") -> pd.Series:
    """The daily analogue of the original's intraday bar range.

    Absolute day-on-day change in the yield, in decimals. Taken from the
    publication-lagged panel, so the observation driving a label on day t was
    knowable on day t.
    """
    return (pit[column].diff().abs() / 100.0).rename("vol_obs")


def markov_regime(
    pit: pd.DataFrame,
    column: str = "DGS10",
    recalibrate_every: int = 252,
    warmup: int = 756,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run the filter across the panel, refitting on expanding history.

    Returns the hard label per day and the full belief matrix. The belief is
    worth keeping: a strategy sized by ``P(high vol)`` behaves very differently
    from one that flips on ``argmax``, and the dashboard shows how often the
    model is actually confident.
    """
    obs = volatility_observation(pit, column)
    values = obs.to_numpy(dtype="float64")
    index = obs.index

    model = MarkovVolatilityRegime()
    calibrated = False
    beliefs = np.full((values.size, 3), np.nan)
    labels: list[str | float] = []

    for i in range(values.size):
        # Refit on everything strictly before today, at the chosen cadence.
        if i >= warmup and (not calibrated or i % recalibrate_every == 0):
            if model.calibrate(values[:i]):
                calibrated = True

        if not calibrated:
            labels.append(np.nan)
            continue

        belief = model.step(values[i])
        beliefs[i] = belief
        labels.append(STATE_LABELS[int(np.argmax(belief))])

    label_series = pd.Series(labels, index=index, dtype="object", name="markov_vol")
    belief_frame = pd.DataFrame(beliefs, index=index, columns=list(STATE_LABELS))
    return label_series, belief_frame


def markov_vol_regime(pit: pd.DataFrame) -> pd.Series:
    """Registry-compatible wrapper returning labels only."""
    labels, _ = markov_regime(pit)
    return labels
