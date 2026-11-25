"""Likelihood field model for range finders -- section 6.4.

Implement ``likelihood_field_model`` to complete task 3.1.  It implements
Table 6.3: project every (non-max-range) beam endpoint into the map frame
of a candidate pose, look up the squared distance to the nearest obstacle
in the pre-computed distance field, and score the beam with a Gaussian
plus a uniform component.  The scan probability is the product over
beams (computed in log space for numerical stability).

Our laser sits exactly at the robot origin and points along the heading,
so the sensor offset ``(x_k_sens, y_k_sens)`` of equation 6.33 is zero and
``theta_k_sens`` is the beam angle relative to the heading.

Notation in the TODO listing follows the book: z_k is the k-th measured
range, theta_k its beam angle relative to the heading, and dist² the
squared distance from an endpoint to the nearest obstacle.
"""

from __future__ import annotations
from .utils import prob_normal

import numpy as np

from .beam_model import BeamParams

# Default intrinsic parameters for the likelihood field model.  They differ
# from the physical laser parameters on purpose: sigma_hit here models the
# 2-D spread of beam endpoints around obstacles (pose error, map error,
# ...) rather than the 1-D range noise of a single beam, so it is much
# broader.  The book's likelihood fields (figure 6.7b) are similarly soft;
# section 6.7 recommends inflating model uncertainty.
LF_PARAMS = BeamParams(
    z_hit=0.6,
    z_short=0.0,
    z_max=0.0,
    z_rand=0.4,
    sigma_hit=0.3,
    lambda_short=1.0,
    z_max_range=8.0,
)


def likelihood_field_model(scan, poses, beam_angles, world, params=None):
    """Scan likelihood p(z_t | x_t, m) under the likelihood field model.

    Parameters
    ----------
    scan : (K,) measured ranges; beams equal to the maximum range are
        discarded, as prescribed by Table 6.3.
    poses : (3,) candidate pose or (N, 3) array of candidate poses.
    beam_angles : (K,) beam directions relative to the robot heading.
    world : the arena (provides the pre-computed distance field).
    params : ``BeamParams`` (reuses z_hit, z_rand, sigma_hit, z_max_range).

    Returns
    -------
    float or (N,) scan probabilities (one per pose).
    """
    params = params or LF_PARAMS
    scan = np.asarray(scan, dtype=float)
    beam_angles = np.asarray(beam_angles, dtype=float)

    single = np.ndim(poses) == 1
    poses = np.atleast_2d(np.asarray(poses, dtype=float))  # (N, 3)

    # TODO 3.1 -- implement Table 6.3 (equation 6.33, lines 4-8)
    # ------------------------------------------------------------------
    # Steps (vectorize over poses and beams together with broadcasting:
    # every quantity below can be an (N, K') array):
    #
    # 1. Keep only the beams whose range is strictly below the maximum
    #    range -- max-range beams carry no endpoint information and are
    #    simply skipped (line 4 of the table).
    # 2. For every kept beam and every candidate pose, transform the beam
    #    endpoint into the map frame (equation 6.33 with zero sensor
    #    offset): the absolute beam direction is the pose heading plus
    #    the beam angle; the endpoint position is the pose position plus
    #    the range times the cosine/sine of that direction.
    # 3. Look up the distance from every endpoint to the nearest obstacle
    #    in the pre-computed field: world.obstacle_distance(x, y) returns
    #    exactly that for arrays of points (points outside the arena come
    #    back far away).  Square the distances (line 7 of the table uses
    #    dist^2).
    # 4. Each beam's probability (line 8) is the mixture of
    #    - a zero-mean Gaussian evaluated at the squared distance with
    #      variance sigma_hit^2 (utils.prob_normal), weighted by z_hit,
    #      and
    #    - the uniform term z_rand / z_max_range.
    # 5. The scan probability per pose is the product over the kept beams
    #    -- sum the logarithms (floor each factor with a tiny constant so
    #    the logarithm stays finite) and exponentiate.
    # 6. Return a float for a single pose, an (N,) array otherwise.
    #                   # skip max-range beams # Max range never hit anyth.
    K = scan < params.z_max_range
    angles=beam_angles[K] # Remove meaningless scans but dont remove poses.
    theta_k = poses[:, 2, None] + angles[None,:]          # absolute beam direction
    x_end = poses[:,0, None]+ scan[None,K]*np.cos(theta_k)                 # eq. 6.33, endpoint in
    y_end = poses[:,1, None] + scan[None,K]*np.sin(theta_k)                 #   the map frame
    dist = world.obstacle_distance(x_end, y_end)                                      # nearest-obstacle lookup, line 7
    p_k = params.z_hit*prob_normal(np.square(dist), np.square(params.sigma_hit)) + params.z_rand / params.z_max_range
    q = np.exp( np.sum(np.log(np.maximum(p_k, 1e-12)), axis=1) )           # product over beams,
                                           #   summed in log space
    
    if single: return float(q[0])
    else: return q
    # ------------------------------------------------------------------
