import numpy as np
import pandas as pd

EPS = 1e-9


def format_tableau(tableau, basis_vars):
    """Format tableau as a Pandas DataFrame for better display"""
    num_vars = tableau.shape[1] - 1
    headers = [f"x{i+1}" for i in range(num_vars)] + ["RHS"]
    index_labels = [f"B{i+1} (x{bv})" for i, bv in enumerate(basis_vars)] + ["Z-Row"]
    return pd.DataFrame(tableau, index=index_labels, columns=headers)


def _pivot(tableau, basis_vars, row, col):
    """Pivot on (row, col) and record the new basic variable (1-indexed)."""
    tableau[row, :] /= tableau[row, col]
    for i in range(tableau.shape[0]):
        if i != row:
            tableau[i, :] -= tableau[i, col] * tableau[row, :]
    basis_vars[row] = col + 1


def _iterate(tableau, basis_vars, allowed_cols):
    """Run simplex iterations (minimisation) until no reduced cost is negative.

    The last row holds reduced costs; its RHS entry holds minus the objective value.
    Bland's rule (lowest index) guarantees termination on degenerate problems.
    """
    while True:
        costs = tableau[-1, :allowed_cols]
        entering = np.where(costs < -EPS)[0]
        if entering.size == 0:
            return
        col = entering[0]

        column = tableau[:-1, col]
        rows = np.where(column > EPS)[0]
        if rows.size == 0:
            raise Exception("Unbounded solution")
        ratios = tableau[rows, -1] / column[rows]
        best = ratios.min()
        ties = rows[np.isclose(ratios, best)]
        row = min(ties, key=lambda r: basis_vars[r])

        _pivot(tableau, basis_vars, row, col)
        print(f"\nTableau after Pivot (Row: {row+1}, Col: {col+1}):")
        print(format_tableau(tableau, basis_vars))


def two_phase_simplex(A, b, c):
    """Minimise c.x subject to A x = b, x >= 0 using the Two-Phase Simplex method.

    Returns the optimal objective value; prints every tableau along the way.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    m, n = A.shape

    # Make every right-hand side non-negative
    neg = b < 0
    A[neg] *= -1
    b[neg] *= -1

    # Phase 1: add artificial variables and minimise their sum
    tableau = np.hstack([A, np.eye(m), b.reshape(-1, 1)])
    tableau = np.vstack([tableau, np.zeros(n + m + 1)])
    tableau[-1, n:n + m] = 1
    basis_vars = list(range(n + 1, n + m + 1))  # artificials start in the basis
    tableau[-1, :] -= tableau[:-1, :].sum(axis=0)  # price out the artificials

    print("\nInitial Tableau - Phase 1:")
    print(format_tableau(tableau, basis_vars))

    _iterate(tableau, basis_vars, n + m)

    if -tableau[-1, -1] > 1e-7:
        raise Exception("Infeasible Solution")

    # Drive any artificial variable that is still basic (at zero level) out of the basis
    keep = list(range(m))
    for r in range(m):
        if basis_vars[r] > n:
            cols = np.where(np.abs(tableau[r, :n]) > EPS)[0]
            if cols.size:
                _pivot(tableau, basis_vars, r, cols[0])
            else:
                keep.remove(r)  # redundant constraint

    print("\nFinal Tableau - Phase 1 (Proceeding to Phase 2):")
    print(format_tableau(tableau, basis_vars))

    # Phase 2: drop artificial columns, restore the real objective, price out the basis
    tableau = np.vstack([tableau[keep, :], tableau[-1:, :]])[:, list(range(n)) + [n + m]]
    basis_vars = [basis_vars[r] for r in keep]
    tableau[-1, :-1] = c
    tableau[-1, -1] = 0
    for i, bv in enumerate(basis_vars):
        tableau[-1, :] -= c[bv - 1] * tableau[i, :]

    print("\nInitial Tableau - Phase 2:")
    print(format_tableau(tableau, basis_vars))

    _iterate(tableau, basis_vars, n)

    print("\nFinal Tableau - Phase 2 (Optimal Solution Found):")
    print(format_tableau(tableau, basis_vars))
    return -tableau[-1, -1]
