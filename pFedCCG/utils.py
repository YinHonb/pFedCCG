import numpy as np

def _normalize_prob_rows(d: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    d = np.clip(d, 0.0, None)
    row_sum = d.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, eps)
    return d / row_sum


def _js_divergence_log2(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)

    mp = m + eps

    mask_p = p > 0
    mask_q = q > 0

    kl_pm = np.sum(p[mask_p] * np.log2(p[mask_p] / mp[mask_p]))
    kl_qm = np.sum(q[mask_q] * np.log2(q[mask_q] / mp[mask_q]))
    return 0.5 * (kl_pm + kl_qm)


def build_S_from_1_minus_js(d: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    d = _normalize_prob_rows(d, eps=eps)
    N = d.shape[0]
    S = np.empty((N, N), dtype=float)

    for i in range(N):
        S[i, i] = 1.0
        for j in range(i + 1, N):
            js = _js_divergence_log2(d[i], d[j], eps=eps)
            s = 1.0 - js
            s = float(np.clip(s, 0.0, 1.0))
            S[i, j] = s
            S[j, i] = s
    return S


def _proj_symmetric_row_sums(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    Xs = 0.5 * (X + X.T)
    N = q.size
    s = Xs.sum(axis=1)
    r = q - s

    c = r.sum() / (2.0 * N)
    a = (r - c) / N
    return Xs + a[:, None] + a[None, :]


def optimize_W_l2(q, d, alpha,
                  max_iters: int = 2000,
                  step: float | None = None,
                  proj_iters: int = 5,
                  tol: float = 1e-10,
                  eps: float = 1e-12,
                  verbose: bool = False):

    q = np.asarray(q, dtype=float).reshape(-1)
    if q.ndim != 1:
        raise ValueError("q must be a 1D array.")
    if np.any(q <= 0):
        raise ValueError("q must be strictly positive.")
    q = q / q.sum()

    d = _normalize_prob_rows(d, eps=eps)
    N = q.size
    if d.shape[0] != N:
        raise ValueError(f"d must have shape (N,M) with N=len(q)={N}, got {d.shape}.")

    if N == 1:
        return np.ones((1, 1), dtype=float)

    S = build_S_from_1_minus_js(d, eps=eps)

    K = np.outer(q, q)

    L = 2.0 * np.max(1.0 / q)
    if step is None:
        step = 0.49 / L

    inv_q = 1.0 / q
    for it in range(max_iters):
        K_old = K.copy()

        grad = 2.0 * (inv_q[:, None] * K - alpha * S)
        K = K - step * grad

        for _ in range(proj_iters):
            K = np.maximum(K, 0.0)                 # C1
            K = _proj_symmetric_row_sums(K, q)     # C2

        rel = np.linalg.norm(K - K_old, ord="fro") / (np.linalg.norm(K_old, ord="fro") + eps)
        if verbose and (it % 100 == 0 or it == max_iters - 1):
            row_err = np.linalg.norm(K.sum(axis=1) - q)
            sym_err = np.linalg.norm(K - K.T, ord="fro")
            minv = K.min()
            print(f"[it={it:4d}] rel={rel:.3e} row_err={row_err:.3e} sym_err={sym_err:.3e} min={minv:.3e}")
        if rel < tol:
            break

    for _ in range(20):
        K = np.maximum(K, 0.0)
        K = _proj_symmetric_row_sums(K, q)

    W = K / q[:, None]

    W = np.maximum(W, 0.0)
    row_sum = W.sum(axis=1, keepdims=True)
    W = W / np.maximum(row_sum, eps)

    return W

