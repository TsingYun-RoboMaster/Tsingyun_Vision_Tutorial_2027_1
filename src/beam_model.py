"""Beam model of range finders -- section 6.3 of Probabilistic Robotics.

Implement the four component densities, the mixture, and the EM calibration
to complete task 2:

* ``p_hit_density`` .. ``p_rand_density`` -- the four error mechanisms of
  equations 6.4-6.12
* ``p_beam``                            -- their mixture, Table 6.1 line 5
* ``learn_intrinsic_parameters``        -- Table 6.2, the EM algorithm that
  recovers the intrinsic parameters from calibration data

``beam_range_finder_model`` (the scan-level product of Table 6.1) is
provided on top of your per-beam density, as is ``solve_lambda_short`` (see
its docstring for why the textbook's closed-form lambda update needs help).

The four error mechanisms (equations 6.4-6.13)
----------------------------------------------
* hit   -- correct range corrupted by Gaussian noise, truncated to
           [0, z_max] and renormalized;
* short -- an unexpected obstacle closer than the true range: exponential
           density truncated to [0, z*];
* max   -- sensor failure, a point mass at z_max;
* rand  -- uniformly random measurement in [0, z_max).

Notation in the TODO listings follows the book: z* is the noise-free range
(ray cast), z_max the sensor's maximum range, sigma = sigma_hit,
lambda = lambda_short, and Phi is the standard normal CDF
(``utils.gaussian_cdf``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import config
from .utils import gaussian_cdf


@dataclass(frozen=True)
class BeamParams:
    """Intrinsic parameters of the beam model (section 6.3.2)."""

    z_hit: float = 0.72
    z_short: float = 0.10
    z_max: float = 0.05        # mixture weight of the failure component
    z_rand: float = 0.13
    sigma_hit: float = 0.06    # meters
    lambda_short: float = 0.28  # 1/meters
    z_max_range: float = 8.0   # meters, the sensor's maximum range

    @classmethod
    def from_laser(cls, laser=config.LASER):
        return cls(
            z_hit=laser.z_hit,
            z_short=laser.z_short,
            z_max=laser.z_max_weight,
            z_rand=laser.z_rand,
            sigma_hit=laser.sigma_hit,
            lambda_short=laser.lambda_short,
            z_max_range=laser.z_max,
        )

    @property
    def weights(self):
        return np.array([self.z_hit, self.z_short, self.z_max, self.z_rand])


# ==========================================================================
# Per-beam density (the heart of Table 6.1)
# ==========================================================================


def p_hit_density(z, z_star, params):
    """Gaussian around the true range, truncated to [0, z_max] (eq. 6.4-6.6)."""
    # Theoratical spot, z_star, measured spot z. How likely is our measured spot givne our ray casted theoraticla spot
    sigma_hit = params.sigma_hit
    z_max = params.z_max_range
    eta = 1 / (gaussian_cdf((z_max - z_star) / sigma_hit) - gaussian_cdf((0 - z_star) / sigma_hit))
    # Do note that gaussain cdf auto uses np for all the values of z_star, so eta is an np (alos note that z is literally just the measureed fella)
    p = eta * (1 / (params.sigma_hit * np.sqrt(2 * np.pi))) * np.exp(-((z - z_star) ** 2) / (2 * params.sigma_hit ** 2))
    p = np.where((z >= 0) & (z <= params.z_max_range),p,0.0)
    return p
    # TODO 2.1a -- the hit component 
    # ------------------------------------------------------------------
    # 1. Evaluate the Gaussian density with mean z_star and standard
    #    deviation sigma_hit at every measurement z.
    # 2. Normalize it over the sensor's interval [0, z_max] (equation
    #    6.6): the normalizer is one over the difference of the standard
    #    normal CDF at (z_max - z_star)/sigma_hit and at
    #    (0 - z_star)/sigma_hit; utils.gaussian_cdf gives you that CDF.
    # 3. The density is zero outside the interval [0, z_max] (equation
    #    6.4).
    #
    # # Pseudocode:
    # #
    # eta = 1 / ( Phi((z_max - z*)/sigma) - Phi((0 - z*)/sigma) )   # eq. 6.6
    # p = eta · (1/(sigma·√(2π))) · exp( -(z - z*)² / (2·sigma²) )
    # return p where 0 ≤ z ≤ z_max, else 0                         # eq. 6.4
    # # ------------------------------------------------------------------
    # raise NotImplementedError("task 2.1: p_hit_density (eq. 6.4-6.6)")


def p_short_density(z, z_star, params):
    """Exponential truncated to [0, z*] (eq. 6.7-6.10)."""
    lambda_short=params.lambda_short
    z, z_star = np.broadcast_arrays(
        np.asarray(z, dtype=float),
        np.asarray(z_star, dtype=float)
    )
    real= z_star>0
    denominator = 1 - np.exp(-lambda_short*z_star)
    eta = 1 / np.where(real, denominator, 1.0) ## NP.where works with 0D scalars, so we dont have to do  z[wtv] anymore
    p = eta * lambda_short * np.exp(-lambda_short * z) ##I keep getting an error cuz z_star[real] will not work when z_star is a float
    p = np.where((0<=z)&(z<=z_star)&(real),p,0.0)
    return p

    # TODO 2.1b -- the short component
    # ------------------------------------------------------------------
    # 1. Evaluate the exponential density with rate lambda_short at every
    #    measurement z (i.e. lambda_short times e to the minus lambda_short*z).
    # 2. Normalize it over [0, z*] (equations 6.8-6.10): the cumulative
    #    mass there is 1 - e^(-lambda_short * z_star), so the normalizer
    #    is its reciprocal.
    # 3. The density is zero outside [0, z*] (equation 6.7); be careful
    #    with z_star = 0 (return zero everywhere).
    #
    # Pseudocode:
    #
    # 1: eta = 1 / (1 - exp(-lambda·z*))                              # eq. 6.10
    # 2: p = eta · lambda · exp(-lambda·z)
    # 3: return p where 0 ≤ z ≤ z*, else 0                            # eq. 6.7
    # 4: if z* ≤ 0, return 0 everywhere (no truncation interval)
    # ------------------------------------------------------------------
    raise NotImplementedError("task 2.1: p_short_density (eq. 6.7-6.10)")


def p_max_density(z, params):
    """Point mass at the maximum range (eq. 6.11)."""
    z_max = params.z_max_range
    return np.where(np.abs(z - z_max) < 1e-6, 1.0, 0.0)
    # TODO 2.1c -- the max component
    # ------------------------------------------------------------------
    # Return one where the measurement equals the maximum range exactly
    # (compare with a tiny absolute tolerance), zero elsewhere.
    # ------------------------------------------------------------------


def p_rand_density(z, params):
    """Uniform over [0, z_max) (eq. 6.12)."""
    z_max = params.z_max_range
    return np.where((z >= 0) & (z < z_max), 1 / z_max, 0.0)
    # TODO 2.1d -- the random component
    # ------------------------------------------------------------------
    # Return one over z_max where the measurement lies inside [0, z_max),
    # zero at z_max and beyond.



def p_beam(z, z_star, params):
    """Mixture density p(z^k_t | z*) for one beam (Table 6.1, line 5).

    ``z`` and ``z_star`` broadcast against each other (scalar, array of
    beams, or measurement grid).
    """
    z_hit=params.z_hit
    z_short=params.z_short
    z_max=params.z_max
    z_rand=params.z_rand
    return (z_hit*p_hit_density(z, z_star,params) + z_short*p_short_density(z, z_star,params) + z_max*p_max_density(z,params)   + z_rand*p_rand_density(z,params))

    # TODO 2.1e -- the mixture (equation 6.13)
    # ------------------------------------------------------------------
    # The beam probability is the weighted sum of the four component
    # densities above, with the mixture weights z_hit, z_short, z_max,
    # z_rand from ``params``.
    #
    # Pseudocode:
    #
    # 1: return z_hit·p_hit(z, z*) + z_short·p_short(z, z*)
    #          + z_max·p_max(z)      + z_rand·p_rand(z)
    # ------------------------------------------------------------------



def beam_range_finder_model(scan, z_star, params):
    """Scan likelihood p(z_t | x_t, m) -- the product of Table 6.1.

    Returns the per-beam probabilities; use ``np.prod`` (or better,
    ``np.exp(np.sum(np.log(...)))``) for the full-scan product.
    """
    return p_beam(np.asarray(scan, dtype=float), np.asarray(z_star, dtype=float), params)


# ==========================================================================
# Learning the intrinsic parameters with EM (Table 6.2)
# ==========================================================================


def solve_lambda_short(e_short, z, z_star, lo=1e-4, hi=1e4, iters=80):
    """ML estimate of ``lambda_short`` for the *truncated* exponential.

    Equation 6.32 in the book, ``sum(e_short) / sum(e_short * z)``, is the
    ML estimate for an untruncated exponential.  Because ``p_short`` is
    truncated to ``[0, z*]`` -- and the normalizer depends on ``lambda`` --
    the exact M-step has no closed form.  This helper solves it by
    bisection on the score equation (provided; not part of the TODO).
    """
    e = np.asarray(e_short, dtype=float)
    z = np.asarray(z, dtype=float)
    zs = np.asarray(z_star, dtype=float)

    def score(lam):
        # d/dlam of the weighted log-likelihood of the truncated exponential
        t = np.clip(lam * zs, 0.0, 500.0)  # exp saturates; avoids overflow
        return float(np.sum(e * (1.0 / lam - z - zs / np.expm1(t))))

    if score(lo) <= 0.0:
        return lo
    if score(hi) >= 0.0:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def learn_intrinsic_parameters(z, z_star, initial=None, max_iter=50, tol=1e-8):
    """Maximum-likelihood intrinsic parameters via EM (Table 6.2).

    Parameters
    ----------
    z, z_star : arrays of measured and noise-free ranges (same length)
    initial   : starting ``BeamParams`` (EM needs a sensible guess)
    max_iter  : hard cap on EM iterations
    tol       : relative log-likelihood improvement that stops the loop

    Returns
    -------
    (BeamParams, history) where history carries the per-iteration
    log-likelihood and parameter snapshots for the convergence plot.
    """
    z = np.asarray(z, dtype=float)
    z_star = np.asarray(z_star, dtype=float)
    params = initial or BeamParams.from_laser(config.LASER)

    history = {"loglik": [], "params": []}
    prev_ll = -np.inf
    for _ in range(max_iter): ## Bsaically repeat until convergence
        # ---- E-step (TODO 2.2a): responsibilities, eq. 6.28-6.29 ------
        # TODO: compute, for every measurement i, the posterior
        # probability e_hit, e_short, e_max, e_rand that it was produced
        # by each mechanism:
        # 1. evaluate your four component densities at every measurement;
        # 2. the normalizer eta is the reciprocal of the weighted sum of
        #    the four densities with the CURRENT mixture weights;
        # 3. each responsibility is the current weight of a mechanism
        #    times its density, times eta (guard the division: a
        #    vanishing total gets a tiny floor).
        #
        # Pseudocode:
        #
        d_hit   = p_hit_density(z, z_star, params)      # the four component
        d_short = p_short_density(z, z_star, params)    # densities under the
        d_max   = p_max_density(z, params)          # CURRENT parameters
        d_rand  = p_rand_density(z, params)
        z_hit=params.z_hit
        z_short=params.z_short
        z_max=params.z_max
        z_rand=params.z_rand
        total = z_hit*d_hit + z_short*d_short + z_max*d_max + z_rand*d_rand #Literally normalize means each over total
        total = np.maximum(total, 1e-12)                    # guard the division
        e_hit   = z_hit*d_hit   / total             # responsibility of
        e_short = z_short*d_short / total           # each mechanism for
        e_max   = z_max*d_max   / total             # each measurement
        e_rand  = z_rand*d_rand  / total

        #now after this i have an np_array for each e

        # ---- M-step (TODO 2.2b): parameter updates, eq. 6.30-6.32 -----
        # TODO: re-estimate all six intrinsic parameters from the
        # responsibilities:
        # 1. each mixture weight is the average of its responsibility
        #    over all measurements (equation 6.30);
        # 2. sigma_hit is the square root of the responsibility-weighted
        #    mean of the squared deviations (z - z_star)^2 (equation
        #    6.31);
        # 3. lambda_short is recovered exactly for the truncated
        #    exponential by the provided solve_lambda_short(e_short, z,
        #    z_star) -- call it.
        # Store the new parameters in a new BeamParams (z_max_range stays
        # as is) and assign it to ``params``.
        #
        # Pseudocode:
        #
        n = len(z)
        z_hit   = (1/n)*np.sum(e_hit)            # mixture weights:
        z_short = (1/n)*np.sum(e_short)               # average
        z_max   = (1/n)*np.sum(e_max)              # responsibility
        z_rand  = (1/n)*np.sum(e_rand)
        sigma = np.sqrt( np.sum(e_hit*np.square(z - z_star)) / np.sum(e_hit))        # eq. 6.31
        lambda_short = solve_lambda_short(e_short, z, z_star)   # exact, provided
        params = BeamParams(z_hit, z_short, z_max, z_rand, sigma,
                               lambda_short, params.z_max_range)

        ll = float(np.sum(np.log(p_beam(z, z_star, params))))
        history["loglik"].append(ll)
        history["params"].append(replace(params))
        if abs(ll - prev_ll) < tol * max(1.0, abs(ll)):
            break
        prev_ll = ll

    return params, history
