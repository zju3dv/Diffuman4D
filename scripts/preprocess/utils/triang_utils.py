import numpy as np
from scipy.optimize import least_squares

INVALID = -1e6


def project_one_point(kp3d, Ks, Ts):
    """kp3d: (3,) in world; Ks:(m,3,3); Ts:(m,4,4) world->cam -> (m,2)"""
    if (kp3d == INVALID).any():
        return np.ones((Ks.shape[0], 2)) * INVALID, np.ones((Ks.shape[0],)) * INVALID

    kp3d_h = np.append(kp3d, 1.0)  # (4,)
    P = Ks @ Ts[:, :3]  # (m,3,4)
    kp2d_h = P @ kp3d_h  # (m,3)
    depth = kp2d_h[:, 2]
    kp2d_h = kp2d_h[:, :2] / (depth[:, None] + 1e-9)
    return kp2d_h, depth


def project_points(kp3d, Ks, Ts, kp3d_score=None):
    projs = [project_one_point(p, Ks, Ts) for p in kp3d]
    kp2d = np.array([p[0] for p in projs], dtype=float).transpose(1, 0, 2)
    kp2d_depth = np.array([p[1] for p in projs], dtype=float).transpose(1, 0)

    if kp3d_score is not None:
        # repeat kp3d_score for each 2d keypoint
        kp2d_score = kp3d_score[None, :].repeat(kp2d.shape[0], axis=0)
    else:
        kp2d_score = None

    # update keypoint score based on face normal and camera normal
    def get_face_normal(kp3d):
        nose, left_eye, right_eye = kp3d[:3]
        eye_mid = (left_eye + right_eye) / 2
        v1 = right_eye - left_eye
        v2 = nose - eye_mid
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        return normal

    # update keypoint score based on face normal
    if kp3d_score is not None:
        face_normal = get_face_normal(kp3d)
        cam_normal = Ts[:, 2, :3]
        face_cam_cos = -np.dot(cam_normal, face_normal)
        face_cam_score = face_cam_cos * 0.5 + 0.5
        kp2d_score[:, :3] *= face_cam_score[:, None]
        kp2d_score[:, 23:91] *= face_cam_score[:, None]

    return kp2d, kp2d_depth, kp2d_score


def triangulate_one_point(Ks, Ts, kp2d, kp2d_score=None, min_views=3, max_views=24, score_thr=0.6):
    """
    Ks, Ts      : (m, 3,3) & (m,4,4)
    kp2d        : (m, 2) valid 2-D observations
    kp2d_score  : (m,) confidence, ∈[0,1]
    returns     : kp3d(3,), kp3d_score, reproj_err
    """
    if kp2d_score is None:
        # set all scores to 1 if not provided
        kp2d_score = np.ones(kp2d.shape[0], dtype=kp2d.dtype)
    if score_thr is not None:
        # filter out views < score_thr
        score_view_thr = None
        if max_views is not None:
            max_views = min(max_views, kp2d.shape[0])
            # use intersection of score_thr and score_view_thr
            score_view_thr = np.percentile(kp2d_score, 100 * (1 - max_views / kp2d.shape[0]))
            score_thr = max(score_thr, score_view_thr)
        mask = kp2d_score >= score_thr
        n_views = mask.sum()
        if n_views < min_views:
            return None, None, n_views
        # filter out invalid views
        Ks = Ks[mask]
        Ts = Ts[mask]
        kp2d = kp2d[mask]
        kp2d_score = kp2d_score[mask]

    # 1. initial linear solution
    A, w = [], []
    for (u, v), K, T, s in zip(kp2d, Ks, Ts, kp2d_score):
        if s <= 0 or u < 0 or v < 0:
            continue
        P = K @ T[:3]
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
        w.extend([s, s])
    A = np.stack(A)  # (2m,4)
    W = np.diag(np.sqrt(w))
    Aw = W @ A
    _, _, Vt = np.linalg.svd(Aw)
    kp3d_lin_h = Vt[-1]
    kp3d_lin = kp3d_lin_h[:3] / (kp3d_lin_h[3] + 1e-9)

    # 2. non-linear reprojection
    # make w per-coord √weight, multiply residual directly
    coord_w = np.repeat(np.sqrt(kp2d_score), 2)  # (2m,)

    def residual(kp3d):
        pred, _ = project_one_point(kp3d, Ks, Ts)
        pred = pred.reshape(-1)  # (2m,)
        res = (pred - kp2d.reshape(-1)) * coord_w
        return res

    # robust optimization
    res_robust = least_squares(
        residual,
        kp3d_lin,
        method="trf",
        loss="huber",
        f_scale=1.0,
        max_nfev=50,
    )
    kp3d = res_robust.x

    # 3. reprojection error
    kp2d_hat, _ = project_one_point(kp3d, Ks, Ts)
    err_px = np.linalg.norm(kp2d_hat - kp2d, axis=1)
    reproj = (err_px * kp2d_score).sum() / (kp2d_score.sum() + 1e-9)

    # # 4. 3d keypoint score
    # kp3d_score = np.sqrt(np.exp(-reproj / 20) * np.percentile(kp2d_score, 75))

    return kp3d, reproj, n_views


def triangulate_points(Ks, Ts, kp2d, kp2d_score=None, min_views=3, score_thr=0.6):
    """
    Ks            : (n, 3, 3)
    Ts            : (n, 4, 4)
    kp2d          : (n, k, 2)
    kp2d_score    : (n, k)
    min_views     : int
    score_thr     : float
    returns       : (k, 3) kp3d, (k,) score, (k,) reproj error
    """
    if kp2d_score is None:
        # set all scores to 1 if not provided
        kp2d_score = np.ones(kp2d.shape[:2], dtype=kp2d.dtype)
    # sanity check
    n, k, _ = kp2d.shape
    if min_views < 3:
        raise ValueError(f"min_views should be at least 3, got {min_views}.")
    if kp2d.shape != (n, k, 2):
        raise ValueError(f"kp2d must have shape (n, k, 2), got {kp2d.shape}")
    if kp2d_score.shape != (n, k):
        raise ValueError(f"kp2d_score must have shape (n, k), got {kp2d_score.shape}")
    if Ks.shape != (n, 3, 3):
        raise ValueError(f"Ks must have shape (n, 3, 3), got {Ks.shape}")
    if Ts.shape != (n, 4, 4):
        raise ValueError(f"Ts must have shape (n, 4, 4), got {Ts.shape}")

    kp3d = np.full((k, 3), INVALID)
    reproj = np.full((k,), INVALID)
    n_views = np.full((k,), INVALID)

    # triangulate each keypoint
    for i in range(k):
        _kp3d, _reproj, _n_views = triangulate_one_point(
            Ks=Ks,
            Ts=Ts,
            kp2d=kp2d[:, i],
            kp2d_score=kp2d_score[:, i],
            min_views=min_views,
            score_thr=score_thr,
        )
        if _kp3d is not None:
            kp3d[i] = _kp3d
        if _reproj is not None:
            reproj[i] = _reproj
        n_views[i] = _n_views

    return kp3d, reproj, n_views
