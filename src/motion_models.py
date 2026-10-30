"""Robot motion models -- chapter 5 of Probabilistic Robotics.

Implement the four functions below to complete task 1:

* ``velocity_motion_model``        -- Table 5.1  p(x_t | u_t, x_{t-1})
* ``sample_velocity_motion_model`` -- Table 5.3  sampling from the velocity model
* ``odometry_motion_model``        -- Table 5.5  p(x_t | u_t, x_{t-1})
* ``sample_odometry_motion_model`` -- Table 5.6  sampling from the odometry model

The map-based variants (Table 5.7) at the bottom are provided: they wrap
your samplers with rejection against the arena occupancy grid.

Conventions
-----------
* A pose is ``(x, y, theta)``; a velocity control is ``(v, omega)``.
* An odometry control ``u_t`` is a pair of poses ``(2, 3)`` measured in the
  robot-internal odometry frame: ``u_t[0] = xbar_{t-1}``, ``u_t[1] = xbar_t``.
* The sampling functions accept a single pose ``(3,)`` or a whole particle
  set ``(N, 3)``; vectorize with numpy for speed.
* ``rng`` is a ``numpy.random.Generator``; draw noise with
  ``utils.sample_normal(variance)`` and evaluate densities with
  ``utils.prob_normal(value, variance)``.
* Notation in the pseudocode listings: a trailing ``_hat`` marks the noisy
  or implied version of a quantity (v_hat = v with a hat in the book);
  ``wrap(a)`` maps an angle into (-pi, pi] (``utils.wrap_angle``);
  ``dist(p, q)`` is the Euclidean distance between two points.
"""

from __future__ import annotations

import numpy as np

from . import config
from .utils import prob_normal, sample_normal, wrap_angle


# ==========================================================================
# Velocity motion model
# ==========================================================================

# util function wrap
# def wrap_angle(alpha):
#     return alpha - 2 * np.pi * np.floor((alpha + np.pi) / (2 * np.pi))


# util function dist
def _dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def velocity_motion_model(x_t, u_t, x_prev, alphas=config.VELOCITY_ALPHAS, dt=config.DT):
    """Probability p(x_t | u_t, x_{t-1}) under the velocity motion model.

    Implements Table 5.1 (equations 5.17-5.32).  ``u_t = (v, omega)``.
    ``x_t`` may be a single pose ``(3,)`` or an array of poses ``(N, 3)``;
    return one probability per pose.

    Reading guide: the algorithm inverts the motion -- it asks "which
    circular arc connects x_prev to the position of x_t?" and reads off the
    velocities that arc implies, then scores how far those implied
    velocities are from the commanded ones under the error model.
    """
    v, w = float(u_t[0]), float(u_t[1])
    a1, a2, a3, a4, a5, a6 = alphas

    x, y, th = (float(x_prev[0]), float(x_prev[1]), float(x_prev[2]))
    x_t = np.asarray(x_t, dtype=float)
    xp, yp, thp = (x_t[0], x_t[1], x_t[2]) if x_t.ndim == 1 else (
        x_t[:, 0], x_t[:, 1], x_t[:, 2])

    # TODO 1.1 -- implement Table 5.1, lines 2-10 (equations 5.17-5.32)
    # ------------------------------------------------------------------
    # Steps (work with arrays element-wise; x_prev components are scalars):
    #
    # 1. Recover the center of the circle that the robot must have turned
    #    on (equations 5.18-5.19):
    #    - the numerator term is (x - xp)*cos(theta) + (y - yp)*sin(theta)
    #    - the denominator term is (y - yp)*cos(theta) - (x - xp)*sin(theta)
    #    - mu is half the numerator divided by the denominator
    #    - the center x* is the midpoint between the two positions, plus
    #      mu times (y - yp); the center y* is the midpoint plus mu times
    #      (xp - x).
    # 2. The turning radius r* is the Euclidean distance from (x, y) to
    #    the center (equation 5.20).
    # 3. The swept angle delta_theta is the difference of two angles: the
    #    angle of the vector from the center to the final position, minus
    #    the angle of the vector from the center to the initial position
    #    (equation 5.21); use the two-argument arctangent (atan2) for both.
    # 4. The implied velocities (equations 5.23-5.24) are
    #    v_hat = delta_theta * r* / dt   and   omega_hat = delta_theta / dt.
    # 5. The implied extra final rotation (equation 5.25) is
    #    gamma_hat = heading_change / dt - omega_hat, where heading_change
    #    is the wrapped difference thp - theta.
    # 6. Degenerate cases -- handle them or the divisions blow up:
    #    - straight motion (denominator near zero, positions distinct):
    #      v_hat is plus or minus (distance moved / dt) with the sign of
    #      the heading-projection of the displacement onto the heading
    #      direction, omega_hat is zero, gamma_hat is
    #      heading_change / dt;
    #    - rotation in place (positions coincide): v_hat is zero and
    #      omega_hat is heading_change / dt.
    # 7. Return the product of three probabilities (equation 5.32), each
    #    given by a zero-mean error density whose variance scales with the
    #    commanded motion (utils.prob_normal(value, variance)):
    #    - the difference between v and v_hat, variance a1*|v| + a2*|w|
    #    - the difference between omega and omega_hat, variance a3*|v| + a4*|w|
    #    - gamma_hat itself (the commanded extra rotation is zero),
    #      variance a5*|v| + a6*|w|
    #
    # Pseudocode:
    #
    # 1:  num = (x - xp)·cos(th) + (y - yp)·sin(th)
    # 2:  den = (y - yp)·cos(th) - (x - xp)·sin(th)
    # 3:  mu  = ½ · num / den                          # eq. 5.18
    # 4:  x_star = (x + xp)/2 + mu·(y - yp)            # eq. 5.19: arc center
    # 5:  y_star = (y + yp)/2 + mu·(xp - x)
    # 6:  r_star = dist((x, y), (x_star, y_star))      # eq. 5.20: radius
    # 7:  d_theta = atan2(yp - y_star, xp - x_star) - atan2(y - y_star, x - x_star)
    # 8:  v_hat = d_theta·r_star / dt                  # eq. 5.24
    # 9:  w_hat = d_theta / dt
    # 10: gamma_hat = wrap(thp - th)/dt - w_hat        # eq. 5.25
    # 11: if |den| ≈ 0 and the two positions differ:   # straight motion
    # 12:     v_hat = sign((xp - x)·cos(th) + (yp - y)·sin(th)) · dist((x, y), (xp, yp))/dt
    # 13:     w_hat = 0 ;  gamma_hat = wrap(thp - th)/dt
    # 14: if the two positions coincide:               # rotation in place
    # 15:     v_hat = 0 ;  w_hat = wrap(thp - th)/dt ;  gamma_hat = 0
    # 16: return prob_normal(v - v_hat, a1·|v| + a2·|w|)          # eq. 5.32
    #          · prob_normal(w - w_hat, a3·|v| + a4·|w|)
    #          · prob_normal(gamma_hat, a5·|v| + a6·|w|)
    # ------------------------------------------------------------------
    num = (x - xp) * np.cos(th) + (y - yp) * np.sin(th)
    den = (y - yp) * np.cos(th) - (x - xp) * np.sin(th)
    mu = 0.5 * num / den  # eq. 5.18
    x_star = (x + xp) / 2 + mu * (y - yp)  # eq. 5.19: arc center
    y_star = (y + yp) / 2 + mu * (xp - x)
    r_star = _dist((x, y), (x_star, y_star))  # eq. 5.20: radius
    d_theta = np.atan2(yp - y_star, xp - x_star) - np.atan2(y - y_star, x - x_star)
    v_hat = d_theta * r_star / dt  # eq. 5.24
    w_hat = d_theta / dt
    gamma_hat = wrap_angle(thp - th) / dt - w_hat  # eq. 5.25

    straight = (np.abs(den) < 1e-6) & ((np.abs(xp - x) > 1e-6) | (np.abs(yp - y) > 1e-6))
    in_place = (np.abs(xp - x) < 1e-6) & (np.abs(yp - y) < 1e-6)

    v_hat = np.where(straight,
                     np.sign((xp - x) * np.cos(th) + (yp - y) * np.sin(th)) * _dist((x, y), (xp, yp)) / dt,
                     v_hat)
    w_hat = np.where(straight, 0, w_hat)
    gamma_hat = np.where(straight, wrap_angle(thp - th) / dt, gamma_hat)

    v_hat = np.where(in_place, 0, v_hat)
    w_hat = np.where(in_place, wrap_angle(thp - th) / dt, w_hat)
    gamma_hat = np.where(in_place, 0, gamma_hat)

    return (prob_normal(v - v_hat, a1 * np.abs(v) + a2 * np.abs(w))
            * prob_normal(w - w_hat, a3 * np.abs(v) + a4 * np.abs(w))
            * prob_normal(gamma_hat, a5 * np.abs(v) + a6 * np.abs(w)))


def sample_velocity_motion_model(u_t, x_prev, alphas=config.VELOCITY_ALPHAS, dt=config.DT, rng=None):
    """Draw pose samples x_t from p(x_t | u_t, x_{t-1}) -- Table 5.3.

    Accepts ``x_prev`` as ``(3,)`` (one pose -> one sample) or ``(N, 3)``
    (a particle set -> one sample per particle).  Return the same shape.
    """
    if rng is None:
        rng = np.random.default_rng()
    v, w = float(u_t[0]), float(u_t[1])
    a1, a2, a3, a4, a5, a6 = alphas

    single = np.ndim(x_prev) == 1
    x_prev = np.atleast_2d(np.asarray(x_prev, dtype=float))
    x, y, th = x_prev[:, 0], x_prev[:, 1], x_prev[:, 2]

    # TODO 1.2 -- implement Table 5.3 (equations 5.10, 5.15, 5.16)
    # ------------------------------------------------------------------
    # 1. Perturb the commanded control (lines 2-4 of the table): draw
    #    v_hat, omega_hat, gamma_hat, each from a zero-mean distribution
    #    whose variance is the corresponding alpha-combination applied to
    #    |v| and |w| (use utils.sample_normal(variance) per particle).
    # 2. Integrate the noisy velocities over dt (lines 5-7, equation
    #    5.16), for every particle at once:
    #    - if omega_hat is so small that dividing by it is unsafe, fall
    #      back to straight-line motion: advance the position by
    #      v_hat * dt along the current heading;
    #    - otherwise the motion is an arc of turning radius v_hat/omega_hat
    #      around the circle center: x' = x - r*sin(theta) +
    #      r*sin(theta + omega_hat*dt), y' = y + r*cos(theta) -
    #      r*cos(theta + omega_hat*dt), with r = v_hat/omega_hat.
    # 3. The final heading is theta + omega_hat*dt plus the extra noisy
    #    final rotation gamma_hat*dt.
    # 4. Stack (x', y', theta') into the output array; return a single
    #    pose (3,) when the input was a single pose.
    #
    # Pseudocode:
    #
    # 1:  v_hat = v + sample_normal(a1·|v| + a2·|w|)      # one draw per particle
    # 2:  w_hat = w + sample_normal(a3·|v| + a4·|w|)
    # 3:  gamma_hat = sample_normal(a5·|v| + a6·|w|)
    # 4:  if |w_hat| < ε (use ε = 1e-6):                  # straight-line fallback
    # 5:      xp = x + v_hat·dt·cos(th)
    # 6:      yp = y + v_hat·dt·sin(th)
    # 7:  else:                                           # arc, eq. 5.16
    # 8:      r = v_hat / w_hat                           # turning radius
    # 9:      xp = x - r·sin(th) + r·sin(th + w_hat·dt)
    # 10:     yp = y + r·cos(th) - r·cos(th + w_hat·dt)
    # 11: thp = th + w_hat·dt + gamma_hat·dt              # extra final rotation
    # 12: return (xp, yp, thp) per input pose
    # ------------------------------------------------------------------
    v_hat = v + sample_normal((a1 * np.abs(v) + a2 * np.abs(w)) * np.ones_like(x), rng)  # one draw per particle
    w_hat = w + sample_normal((a3 * np.abs(v) + a4 * np.abs(w)) * np.ones_like(x), rng)
    gamma_hat = sample_normal((a5 * np.abs(v) + a6 * np.abs(w)) * np.ones_like(x), rng)

    straight = np.abs(w_hat) < 1e-6
    r = np.where(straight, 1.0, v_hat / np.where(straight, 1.0, w_hat))  # turning radius
    xp = x - r * np.sin(th) + r * np.sin(th + w_hat * dt)
    yp = y + r * np.cos(th) - r * np.cos(th + w_hat * dt)

    xp = np.where(straight, x + v_hat * dt * np.cos(th), xp)
    yp = np.where(straight, y + v_hat * dt * np.sin(th), yp)

    thp = th + w_hat * dt + gamma_hat * dt  # extra final rotation
    poses = np.stack([xp, yp, thp], axis=1)
    return poses[0] if single else poses


# ==========================================================================
# Odometry motion model
# ==========================================================================


def odometry_motion_model(x_t, u_t, x_prev, alphas=config.ODOMETRY_ALPHAS):
    """Probability p(x_t | u_t, x_{t-1}) under the odometry model.

    Implements Table 5.5 (equations 5.34-5.46).  ``u_t`` is the odometry
    pose pair ``(2, 3)``.  ``x_t`` may be a pose or an array of poses.
    """
    a1, a2, a3, a4 = alphas
    xb, yb, thb = u_t[0]
    xbp, ybp, thbp = u_t[1]

    x, y, th = (float(x_prev[0]), float(x_prev[1]), float(x_prev[2]))
    x_t = np.asarray(x_t, dtype=float)
    xp, yp, thp = (x_t[0], x_t[1], x_t[2]) if x_t.ndim == 1 else (
        x_t[:, 0], x_t[:, 1], x_t[:, 2])

    # TODO 1.3 -- implement Table 5.5, lines 2-11 (equations 5.34-5.46)
    # ------------------------------------------------------------------
    # 1. Decompose the odometry pose pair into its relative motion
    #    (equations 5.34-5.36):
    #    - rot1 is the angle of the displacement between the two odometry
    #      positions (atan2 of the y and x differences) minus the initial
    #      odometry heading, wrapped to [-pi, pi];
    #    - trans is the Euclidean distance between the odometry positions;
    #    - rot2 is the wrapped odometry heading change minus rot1.
    # 2. Decompose the hypothetical motion from x_prev to x_t the same
    #    way (lines 5-7): hat_rot1, hat_trans, hat_rot2.
    # 3. Three independent error probabilities (lines 8-10, equations
    #    5.44-5.46): each is a zero-mean density over the *wrapped*
    #    difference between the odometry value and the hypothesized value:
    #    - rot1 difference, variance a1*|hat_rot1| + a2*hat_trans
    #    - trans difference, variance a3*hat_trans + a4*(|hat_rot1| + |hat_rot2|)
    #    - rot2 difference, variance a1*|hat_rot2| + a2*hat_trans
    #    (use utils.prob_normal and utils.wrap_angle)
    # 4. Return the product of the three probabilities.
    #
    # Pseudocode:
    #
    # relative motion measured by the odometry (eqs. 5.34-5.36):
    # 1:  rot1  = wrap(atan2(ybp - yb, xbp - xb) - thb)
    # 2:  trans = dist((xb, yb), (xbp, ybp))
    # 3:  rot2  = wrap(thbp - thb - rot1)
    # relative motion implied by the hypothetical pose pair (lines 5-7):
    # 4:  rot1_hat  = wrap(atan2(yp - y, xp - x) - th)
    # 5:  trans_hat = dist((x, y), (xp, yp))
    # 6:  rot2_hat  = wrap(thp - th - rot1_hat)
    # error probabilities (lines 8-10, eqs. 5.44-5.46; wrap each difference!):
    # 7:  p1 = prob_normal(wrap(rot1 - rot1_hat),   a1·|rot1_hat| + a2·trans_hat)
    # 8:  p2 = prob_normal(wrap(trans - trans_hat), a3·trans_hat + a4·(|rot1_hat| + |rot2_hat|))
    # 9:  p3 = prob_normal(wrap(rot2 - rot2_hat),   a1·|rot2_hat| + a2·trans_hat)
    # 10: return p1 · p2 · p3
    # ------------------------------------------------------------------
    rot1 = wrap_angle(np.atan2(ybp - yb, xbp - xb) - thb)
    trans = np.sqrt((xb - xbp) ** 2 + (yb - ybp) ** 2)
    rot2 = wrap_angle(thbp - thb - rot1)
    rot1_hat = wrap_angle(np.atan2(yp - y, xp - x) - th)
    trans_hat = _dist((x, y), (xp, yp))
    rot2_hat = wrap_angle(thp - th - rot1_hat)
    p1 = prob_normal(wrap_angle(rot1 - rot1_hat), a1 * np.abs(rot1_hat) + a2 * trans_hat)
    p2 = prob_normal(wrap_angle(trans - trans_hat), a3 * trans_hat + a4 * (np.abs(rot1_hat) + np.abs(rot2_hat)))
    p3 = prob_normal(wrap_angle(rot2 - rot2_hat), a1 * np.abs(rot2_hat) + a2 * trans_hat)
    return p1 * p2 * p3


def sample_odometry_motion_model(u_t, x_prev, alphas=config.ODOMETRY_ALPHAS, rng=None):
    """Draw pose samples x_t from p(x_t | u_t, x_{t-1}) -- Table 5.6."""
    if rng is None:
        rng = np.random.default_rng()
    a1, a2, a3, a4 = alphas
    xb, yb, thb = u_t[0]
    xbp, ybp, thbp = u_t[1]

    single = np.ndim(x_prev) == 1
    x_prev = np.atleast_2d(np.asarray(x_prev, dtype=float))
    x, y, th = x_prev[:, 0], x_prev[:, 1], x_prev[:, 2]

    # TODO 1.4 -- implement Table 5.6 (equations 5.37-5.40)
    # ------------------------------------------------------------------
    # 1. Decompose the odometry pose pair into rot1, trans, rot2 exactly
    #    as in task 1.3 (equations 5.34-5.36; the book repeats these lines
    #    in Table 5.6).
    # 2. Perturb the relative motion (lines 5-7, equations 5.37-5.39):
    #    subtract, from each of rot1, trans, rot2, a zero-mean noise draw
    #    whose variance is
    #    - for hat_rot1: a1*|rot1| + a2*trans
    #    - for hat_trans: a3*trans + a4*(|rot1| + |rot2|)
    #    - for hat_rot2: a1*|rot2| + a2*trans
    #    (use utils.sample_normal per particle)
    # 3. Compose the noisy relative motion onto every particle pose
    #    (lines 8-10, equation 5.40): first rotate by hat_rot1, then
    #    translate by hat_trans along the new heading, then rotate again
    #    by hat_rot2.
    # 4. Stack the resulting poses; return a single pose for single-pose
    #    input.
    #
    # Pseudocode:
    #
    # relative motion measured by the odometry (eqs. 5.34-5.36):
    # 1:  rot1  = wrap(atan2(ybp - yb, xbp - xb) - thb)
    # 2:  trans = dist((xb, yb), (xbp, ybp))
    # 3:  rot2  = wrap(thbp - thb - rot1)
    # noisy relative motion, one draw per particle (eqs. 5.37-5.39):
    # 4:  rot1_hat  = rot1  - sample_normal(a1·|rot1|  + a2·trans)
    # 5:  trans_hat = trans - sample_normal(a3·trans + a4·(|rot1| + |rot2|))
    # 6:  rot2_hat  = rot2  - sample_normal(a1·|rot2|  + a2·trans)
    # compose onto each particle pose (eq. 5.40):
    # 7:  xp = x + trans_hat·cos(th + rot1_hat)
    # 8:  yp = y + trans_hat·sin(th + rot1_hat)
    # 9:  thp = th + rot1_hat + rot2_hat
    # 10: return (xp, yp, thp) per input pose
    # ------------------------------------------------------------------
    rot1 = wrap_angle(np.atan2(ybp - yb, xbp - xb) - thb)
    trans = _dist((xb, yb), (xbp, ybp))
    rot2 = wrap_angle(thbp - thb - rot1)
    rot1_hat = rot1 - sample_normal((a1 * np.abs(rot1) + a2 * trans) * np.ones_like(x), rng)
    trans_hat = trans - sample_normal((a3 * trans + a4 * (np.abs(rot1) + np.abs(rot2))) * np.ones_like(x), rng)
    rot2_hat = rot2 - sample_normal((a1 * np.abs(rot2) + a2 * trans) * np.ones_like(x), rng)
    xp = x + trans_hat * np.cos(th + rot1_hat)
    yp = y + trans_hat * np.sin(th + rot1_hat)
    thp = th + rot1_hat + rot2_hat

    poses = np.stack([xp, yp, thp], axis=1)
    return poses[0] if single else poses


# ==========================================================================
# Map-based motion models (Table 5.7) -- provided
# ==========================================================================


def _rejection_sample(sampler, world, clearance, rng, fallback, max_rounds=50):
    """Redraw samples until every pose lies in free space (Table 5.7).

    If the commanded motion pushes a particle into a wall from every
    direction, the particle stays put after ``max_rounds`` rounds -- like
    the real robot, it bumped into the obstacle instead of moving through.
    """
    poses = sampler(rng)
    pending = ~world.is_free(poses[:, 0], poses[:, 1], clearance)
    rounds = 0
    while pending.any() and rounds < max_rounds:
        redrawn = sampler(rng)
        poses[pending] = redrawn[pending]
        pending = ~world.is_free(poses[:, 0], poses[:, 1], clearance)
        rounds += 1
    if pending.any():
        poses[pending] = fallback[pending]
    return poses


def sample_velocity_motion_model_with_map(u_t, x_prev, world, alphas=config.VELOCITY_ALPHAS,
                                          dt=config.DT, rng=None, clearance=0.0):
    """Table 5.7: velocity-model samples that avoid occupied space."""
    single = np.ndim(x_prev) == 1
    x_prev = np.atleast_2d(np.asarray(x_prev, dtype=float))
    sampler = lambda r: sample_velocity_motion_model(u_t, x_prev, alphas, dt, r)
    poses = _rejection_sample(sampler, world, clearance, rng or np.random.default_rng(), x_prev)
    return poses[0] if single else poses


def sample_odometry_motion_model_with_map(u_t, x_prev, world, alphas=config.ODOMETRY_ALPHAS,
                                          rng=None, clearance=0.0):
    """Table 5.7: odometry-model samples that avoid occupied space."""
    single = np.ndim(x_prev) == 1
    x_prev = np.atleast_2d(np.asarray(x_prev, dtype=float))
    sampler = lambda r: sample_odometry_motion_model(u_t, x_prev, alphas, r)
    poses = _rejection_sample(sampler, world, clearance, rng or np.random.default_rng(), x_prev)
    return poses[0] if single else poses
