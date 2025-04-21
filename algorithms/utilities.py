import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import cross_val_score
from typing import Tuple


def subset_accuracy(p, sw) -> Tuple[float, float]:
    """
    Computes a weighted fitness function for feature selection using:
    - f1: model performance (recall)
    - f2: relevance via RDC (non-linear dependency)
    - f3: selection stability (feature frequency)

    Returns:
        (1 - final_score, f1) where f1 is recall and (1 - score) is cost
    """
    subset = sw.X[:, p.b == 1]

    if p._nbf == 0:
        return 1.0, 0.0

    f1 = cross_val_score(sw.clf, subset, sw.y, cv=sw.cv, scoring='recall', n_jobs=-1).mean()
    f2 = np.sum(sw.irdc[p.b == 1]) / p._nbf

    max_freq = np.max(sw.freq)
    if max_freq > 1:
        norm_freq = [(f - 1) / (max_freq - 1) for f in sw.freq[p.b == 1]]
        f3 = np.sum(norm_freq) / p._nbf
    else:
        f3 = 0.0

    final_score = sw.alpha_1 * f1 + sw.alpha_2 * f2 + sw.alpha_3 * f3
    return 1 - final_score, f1


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid function for PSO binary encoding.
    """
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU binary activation (not currently used).
    """
    return np.array([1 if i > 0 else 0 for i in x])


def RDC(x, y, f=np.sin, k=20, s=1/6., n=10) -> float:
    """
    Computes the Randomized Dependence Coefficient (RDC).

    Args:
        x, y: input arrays (1D or 2D)
        f: non-linear projection function (default: sin)
        k: number of projections
        s: scale factor
        n: number of times to repeat for stability

    Returns:
        RDC score [0,1]
    """
    if n > 1:
        values = []
        for _ in range(n):
            try:
                values.append(RDC(x, y, f, k, s, 1))
            except np.linalg.LinAlgError:
                continue
        return np.median(values)

    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y

    # Copula transformation
    cx = np.column_stack([rankdata(col, method='ordinal') for col in x.T]) / float(x.shape[0])
    cy = np.column_stack([rankdata(col, method='ordinal') for col in y.T]) / float(y.shape[0])

    X = np.column_stack([cx, np.ones(cx.shape[0])])
    Y = np.column_stack([cy, np.ones(cy.shape[0])])

    Rx = (s / X.shape[1]) * np.random.randn(X.shape[1], k)
    Ry = (s / Y.shape[1]) * np.random.randn(Y.shape[1], k)

    fX = f(np.dot(X, Rx))
    fY = f(np.dot(Y, Ry))

    C = np.cov(np.hstack([fX, fY]).T)

    k0 = k
    lb = 1
    ub = k

    while True:
        Cxx = C[:k, :k]
        Cyy = C[k0:k0 + k, k0:k0 + k]
        Cxy = C[:k, k0:k0 + k]
        Cyx = C[k0:k0 + k, :k]

        try:
            eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                            np.dot(np.linalg.pinv(Cyy), Cyx)))
        except np.linalg.LinAlgError:
            return 0.0

        if not (np.all(np.isreal(eigs)) and 0 <= np.min(eigs) <= 1 and np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue

        if lb == ub:
            break
        lb = k
        k = ub if ub == lb + 1 else (ub + lb) // 2

    return float(np.sqrt(np.max(eigs)))
