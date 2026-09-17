"""Small educational SVD companion based on Jacobi eigendecomposition.

This is intentionally separate from the production solver.  It is useful for
showing why the smallest right singular vector solves the homogeneous DLT
system, while NumPy's SVD remains the reliable default for image processing.
"""

from __future__ import annotations

import numpy as np


def jacobi_eigh(matrix: np.ndarray, *, tolerance: float = 1e-12, max_iterations: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues/eigenvectors of a real symmetric matrix."""

    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Jacobi eigendecomposition requires a square matrix")
    if not np.allclose(a, a.T, atol=1e-12):
        raise ValueError("Jacobi eigendecomposition requires a symmetric matrix")
    a = a.copy()
    vectors = np.eye(a.shape[0], dtype=np.float64)

    for _ in range(max_iterations):
        off_diagonal = np.triu(np.abs(a), 1)
        p, q = np.unravel_index(np.argmax(off_diagonal), off_diagonal.shape)
        if off_diagonal[p, q] <= tolerance:
            break
        if abs(a[p, p] - a[q, q]) <= tolerance:
            angle = np.pi / 4.0 if a[p, q] > 0 else -np.pi / 4.0
        else:
            angle = 0.5 * np.arctan2(2.0 * a[p, q], a[p, p] - a[q, q])
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.eye(a.shape[0])
        rotation[p, p] = cosine
        rotation[q, q] = cosine
        rotation[p, q] = -sine
        rotation[q, p] = sine
        a = rotation.T @ a @ rotation
        vectors = vectors @ rotation
    else:
        raise RuntimeError("Jacobi eigendecomposition did not converge")

    order = np.argsort(np.diag(a))[::-1]
    return np.diag(a)[order], vectors[:, order]


def jacobi_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an educational full right-singular-vector decomposition."""

    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    eigenvalues, v = jacobi_eigh(a.T @ a)
    singular_values = np.sqrt(np.clip(eigenvalues, 0.0, None))
    u = np.zeros((a.shape[0], a.shape[1]), dtype=np.float64)
    nonzero = singular_values > 1e-14
    u[:, nonzero] = (a @ v[:, nonzero]) / singular_values[nonzero]
    return u, singular_values, v.T


def smallest_right_singular_vector(matrix: np.ndarray) -> np.ndarray:
    """Return the last right singular vector from the educational implementation."""

    return jacobi_svd(matrix)[2][-1]
