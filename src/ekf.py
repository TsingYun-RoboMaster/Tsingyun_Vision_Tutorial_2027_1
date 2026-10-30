"""EKF localization with the velocity motion model -- chapter 7.

Implement ``ekf_predict`` and ``ekf_correct`` to complete task 4.  They
implement Table 7.2 of the book (EKF localization) for our arena:

* ``ekf_predict``  -- lines 2-4: propagate the Gaussian through the
  (velocity) motion model, linearized by G_t, with motion noise R_t.
  The noise-free successor pose is exactly your task-1.2 sampler run
  with zero alphas -- call it instead of reimplementing the arc.
* ``ekf_correct`` -- lines 12-15: fold in a measurement with the Kalman
  gain.  The book loops over point features; our "features" are laser
  beams, so the per-feature sum of updates collapses into one vector
  update with a stacked residual and Jacobian (the book's information-
  space argument, section 7.5.2, says the two forms are equivalent).

Everything else here is provided: the measurement adapters turn a laser
scan into (residual, Jacobian, noise) pairs for the beam model (ray-cast
prediction, numerical Jacobian) and for the likelihood field (endpoint
distance to the nearest obstacle, numerical Jacobian), and
``EKFLocalizer`` wires predict + correct into a running filter.
"""

from __future__ import annotations

import numpy as np

from . import config, utils
from .beam_model import BeamParams
from .motion_models import sample_velocity_motion_model
from .utils import make_rng, wrap_angle


# ==========================================================================
# The EKF step (Table 7.2) -- student implementations
# ==========================================================================


def ekf_predict(mu, Sigma, u_t, alphas=config.VELOCITY_ALPHAS, dt=config.DT):
    """Motion update, Table 7.2 lines 2-4 (velocity model, eq. 7.4/7.8).

    Parameters
    ----------
    mu : (3,) pose mean;  Sigma : (3, 3) pose covariance
    u_t : (v, omega) commanded velocities
    alphas : the six velocity-motion noise parameters of chapter 5
    dt : control duration

    Returns
    -------
    (mu_bar, Sigma_bar) -- the predicted Gaussian.
    """
    v, w = float(u_t[0]), float(u_t[1])
    a1, a2, a3, a4, a5, a6 = alphas

    # TODO 4.1 -- implement Table 7.2, lines 2-4 (equations 7.4, 7.8)
    # ------------------------------------------------------------------
    # 1. Propagate the mean through the noise-free motion (line 2,
    #    equation 7.4): the arc of turning radius v/omega, or the
    #    straight line when omega is ~0.  Do not reimplement it -- your
    #    task-1.2 sampler with all alphas set to zero performs zero noise
    #    draws, so calling it yields exactly the noise-free successor
    #    pose.
    # 2. Build G_t, the 3x3 Jacobian of the motion map with respect to
    #    the pose (equation 7.8): identity except for the theta column,
    #    whose entries combine the turning radius with sines/cosines of
    #    theta and of theta + omega*dt.
    # 3. Build R_t, the motion noise covariance.  The book leaves its
    #    design "as an exercise" (p. 172); use the chapter-5 noise model
    #    so the EKF matches your task-1 sampler:
    #    - V is the Jacobian of the motion map with respect to the noisy
    #      controls (v_hat, omega_hat, gamma_hat);
    #    - M is the diagonal matrix of their variances, the same
    #      alpha-combinations of |v| and |omega| as in Table 5.3;
    #    - R_t = V M V^T.
    # 4. The predicted covariance is G Sigma G^T + R_t (line 4).
    #
    # Pseudocode:
    #
    # 1:  mu_bar = sample_velocity_motion_model(u_t, mu, (0,...,0), dt)
    # 2:  w_safe = omega if |omega| >= 1e-4 else sign(omega)*1e-4
    # 3:  r = v / w_safe
    # 4:  G = [[1, 0, -r*cos(th) + r*cos(th + w_safe*dt)],
    #         [0, 1, -r*sin(th) + r*sin(th + w_safe*dt)],
    #         [0, 0, 1]]
    # 5:  s = sin(th) ;  c = cos(th)
    # 6:  sp = sin(th + w_safe*dt) ;  cp = cos(th + w_safe*dt)
    # 7:  V = [[(-s + sp)/w_safe ,  v*(s - sp)/w_safe^2 + v*cp*dt/w_safe ,  0],
    #         [( c - cp)/w_safe ,  v*(c - cp)/w_safe^2 + v*sp*dt/w_safe ,  0],
    #         [0               ,  dt                                  ,  dt]]
    # 8:  M = diag(a1*|v| + a2*|w| ,  a3*|v| + a4*|w| ,  a5*|v| + a6*|w|)
    # 9:  Sigma_bar = G*Sigma*G^T + V*M*V^T
    # 10: return (mu_bar, Sigma_bar)
    # ------------------------------------------------------------------
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    mu_bar = np.asarray(sample_velocity_motion_model(u_t, mu, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dt), dtype=float)
    w_safe = w if abs(w) > 1e-4 else (1e-4 if w >= 0.0 else -1e-4)
    r = v / w_safe

    th = float(mu[2])
    s, c = np.sin(th), np.cos(th)
    sp, cp = np.sin(th + w_safe * dt), np.cos(th + w_safe * dt)
    G = np.array([
        [1.0, 0.0, -r * c + r * cp],
        [0.0, 1.0, -r * s + r * sp],
        [0.0, 0.0, 1.0],
    ])
    V = np.array([
        [(-s + sp) / w_safe, v * (s - sp) / (w_safe ** 2) + v * cp * dt / w_safe, 0],
        [(c - cp) / w_safe, v * (c - cp) / (w_safe ** 2) + v * sp * dt / w_safe, 0],
        [0, dt, dt],
    ])
    M = np.diag([
        a1 * abs(v) + a2 * abs(w),
        a3 * abs(v) + a4 * abs(w),
        a5 * abs(v) + a6 * abs(w),
    ])
    Sigma_bar = G @ Sigma @ G.T + V @ M @ V.T
    return mu_bar, Sigma_bar


def ekf_correct(mu_bar, Sigma_bar, residual, H, R):
    """Measurement update, Table 7.2 lines 12-15 (stacked-beam form).

    Parameters
    ----------
    mu_bar, Sigma_bar : the predicted Gaussian from ``ekf_predict``
    residual : (K,) innovation vector z - h(mu_bar)
    H : (K, 3) Jacobian of the measurement map at mu_bar
    R : (K, K) diagonal measurement-noise covariance

    Returns
    -------
    (mu, Sigma) -- the corrected Gaussian.
    """
    # TODO 4.2 -- implement Table 7.2, lines 12-15 (stacked-beam form)
    # ------------------------------------------------------------------
    # 1. The innovation covariance S measures how much the residual is
    #    expected to vary: the Jacobian-propagated pose uncertainty plus
    #    the measurement noise (line 12).
    # 2. The Kalman gain K (line 12) weighs belief against measurement:
    #    how much of the residual to fold back into the pose.
    # 3. The corrected mean is mu_bar plus K times the residual (line 14);
    #    wrap the heading component back into (-pi, pi].
    # 4. The corrected covariance shrinks by what was learned (line 15).
    #
    # Pseudocode:
    #
    # 1: S = H*Sigma_bar*H^T + R
    # 2: K = Sigma_bar*H^T*S^-1
    # 3: mu = mu_bar + K*residual ;  mu[theta] = wrap(mu[theta])
    # 4: Sigma = (I - K*H)*Sigma_bar
    # 5: return (mu, Sigma)
    # ------------------------------------------------------------------
    mu_bar = np.asarray(mu_bar, dtype=float)
    Sigma_bar = np.asarray(Sigma_bar, dtype=float)
    residual = np.asarray(residual, dtype=float)
    H = np.asarray(H, dtype=float)
    R = np.asarray(R, dtype=float)

    S = H @ Sigma_bar @ H.T + R
    K = Sigma_bar @ H.T @ np.linalg.inv(S)
    mu = mu_bar + K @ residual
    mu[2] = wrap_angle(mu[2])
    Sigma = (np.eye(3) - K @ H) @ Sigma_bar
    return mu, Sigma


# ==========================================================================
# Measurement adapters (provided)
# ==========================================================================


def _numeric_pose_jacobian(fn, pose, eps=(0.05, 0.05, 0.02)):
    """Columns of the Jacobian of ``fn(pose) -> vector`` w.r.t. (x, y, th),
    by central differences."""
    base = fn(pose)
    J = np.empty((len(base), 3))
    for j, e in enumerate(eps):
        plus, minus = pose.copy(), pose.copy()
        plus[j] += e
        minus[j] -= e
        J[:, j] = (fn(plus) - fn(minus)) / (2.0 * e)
    return J


def beam_update_terms(world, pose_bar, scan, beam_angles, params=None):
    """Residual/Jacobian/noise for a beam-model update (provided).

    The predicted measurement h(x) is the noise-free scan obtained by ray
    casting from the pose (Table 6.1, line 4); the residual is the range
    difference on informative beams.  Max-range beams and gated outliers
    (likely short/random readings) are dropped.
    """
    params = params or BeamParams.from_laser()
    ec = config.EKF
    idx = np.arange(0, len(scan), ec.beam_subsample)
    z = np.asarray(scan)[idx]
    ang = np.asarray(beam_angles)[idx]
    keep = (z > 0.05) & (z < params.z_max_range - 1e-9)
    z, ang = z[keep], ang[keep]

    def z_star(p):
        return world.raycast([[p[0], p[1]]], p[2] + ang, params.z_max_range)[0]

    zs = z_star(pose_bar)
    good = zs < params.z_max_range - 1e-6  # no prediction for misses
    good &= np.abs(z - zs) < ec.beam_gate  # outlier gating
    if good.sum() < 6:
        return None

    # H is the derivative of the predicted measurement z*(x) w.r.t. the
    # pose, matching the residual z - z*(mu_bar); drop beams whose
    # linearization is unreliable -- near-grazing rays make the ray-cast
    # range sharply discontinuous in the pose, so their Jacobian rows
    # explode
    H_full = _numeric_pose_jacobian(z_star, pose_bar)
    good &= np.linalg.norm(H_full, axis=1) < ec.beam_jacobian_gate
    if good.sum() < 6:
        return None

    residual = (z - zs)[good]
    H = H_full[good]
    # inflated per-beam noise: the mixture has heavy tails (short/random
    # readings), so an honest R is much larger than sigma_hit^2 alone
    R = np.diag(np.full(int(good.sum()), ec.beam_noise_var))
    info = {"kind": "rays", "angles": ang[good], "z": z[good]}
    return residual, H, R, info


def likelihood_field_update_terms(world, pose_bar, scan, beam_angles, params=None):
    """Residual/Jacobian/noise for a likelihood-field update (provided).

    The measurement map h(x) evaluates, for every beam endpoint projected
    from the pose, its distance to the nearest obstacle in the
    pre-computed field -- at the true pose every endpoint sits on an
    obstacle, so the target value is zero.  Beams whose endpoints land far
    from any obstacle (short/random readings) are gated out.
    """
    from .likelihood_field import LF_PARAMS

    params = params or LF_PARAMS
    z = np.asarray(scan, dtype=float)
    ang = np.asarray(beam_angles, dtype=float)
    keep = z < params.z_max_range - 1e-9
    z, ang = z[keep], ang[keep]

    def h(p):
        th = p[2] + ang
        return world.obstacle_distance(p[0] + z * np.cos(th), p[1] + z * np.sin(th))

    h0 = h(pose_bar)
    good = h0 < config.EKF.field_gate
    if good.sum() < 6:
        return None

    residual = -h0[good]  # target 0 minus h(mu_bar)
    H = _numeric_pose_jacobian(h, pose_bar)[good]
    R = np.diag(np.full(int(good.sum()), params.sigma_hit ** 2 + 0.02))
    info = {"kind": "endpoints", "angles": ang[good], "z": z[good]}
    return residual, H, R, info


def load_beam_params():
    """Physically calibrated beam parameters, or the EM-learned ones if
    task 2 already wrote them to data/."""
    import json

    if config.LEARNED_PARAMS_PATH.exists():
        p = json.loads(config.LEARNED_PARAMS_PATH.read_text())
        return BeamParams(
            z_hit=p["z_hit"], z_short=p["z_short"], z_max=p["z_max"],
            z_rand=p["z_rand"], sigma_hit=p["sigma_hit"],
            lambda_short=p["lambda_short"], z_max_range=p["z_max_range"],
        )
    return BeamParams.from_laser()


# ==========================================================================
# The running filter (provided glue)
# ==========================================================================


class EKFLocalizer:
    """One EKF localization instance driven by the velocity motion model.

    Bookkeeping for the speed-vs-accuracy story of chapter 6: every
    measurement update is timed and remembers which beams it used, so
    the demos can show what each sensor model costs.
    """

    def __init__(self, world, sensor="beam", beam_params=None,
                 init_pose=None, seed=None):
        self.world = world
        self.sensor = sensor
        self.beam_params = beam_params or load_beam_params()
        rng = make_rng(seed)
        std = np.array(config.EKF.init_std)
        self.mu = (np.asarray(init_pose, dtype=float)
                   + rng.normal(0.0, std * 0.5))
        self.mu[2] = wrap_angle(self.mu[2])
        self.Sigma = np.diag(std ** 2)
        self.update_ms_total = 0.0
        self.update_count = 0
        self.last_update_info = None

    def predict(self, u_t):
        """Motion update with the commanded velocity control."""
        self.mu, self.Sigma = ekf_predict(self.mu, self.Sigma, u_t)

    @property
    def mean_update_ms(self):
        return self.update_ms_total / max(self.update_count, 1)

    def update(self, scan, beam_angles):
        """Measurement update with the configured sensor model.

        Timed wall-to-wall (prediction terms + Kalman correction), so the
        HUD can show what each sensor model really costs per update.
        """
        import time

        t0 = time.perf_counter()
        if self.sensor == "beam":
            terms = beam_update_terms(self.world, self.mu, scan,
                                      beam_angles, self.beam_params)
        elif self.sensor == "likelihood":
            terms = likelihood_field_update_terms(self.world, self.mu, scan,
                                                  beam_angles)
        else:
            raise ValueError(f"unknown sensor model: {self.sensor}")
        if terms is not None:
            residual, H, R, info = terms
            self.mu, self.Sigma = ekf_correct(self.mu, self.Sigma, residual, H, R)
            self.update_ms_total += (time.perf_counter() - t0) * 1e3
            self.update_count += 1
            self.last_update_info = info

    def error(self, true_pose):
        pos = float(np.hypot(self.mu[0] - true_pose[0], self.mu[1] - true_pose[1]))
        head = float(abs(wrap_angle(self.mu[2] - true_pose[2])))
        return pos, head
