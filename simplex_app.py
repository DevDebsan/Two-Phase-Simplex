import numpy as np
import pandas as pd

def format_tableau(tableau, basis_vars):
    """Format tableau as a Pandas DataFrame for better display"""
    num_vars = tableau.shape[1] - 1
    headers = [f"x{i+1}" for i in range(num_vars)] + ["RHS"]
    index_labels = [f"B{i+1} (x{bv})" for i, bv in enumerate(basis_vars)] + ["Z-Row"]
    return pd.DataFrame(tableau, index=index_labels, columns=headers)

def two_phase_simplex(A, b, c):
    """Solve the Two-Phase Simplex method with tabular format"""
    m, n = A.shape
    
    # Phase 1: Introduce artificial variables
    A = np.hstack([A, np.eye(m)])
    c_phase1 = np.hstack([np.zeros(n), np.ones(m)])
    tableau = np.hstack([A, b.reshape(-1, 1)])
    tableau = np.vstack([tableau, np.hstack([c_phase1, [0]])])
    basis_vars = list(range(n+1, n+m+1))  # Artificial variables as initial basis

    print("\nInitial Tableau - Phase 1:")
    print(format_tableau(tableau, basis_vars))

    # Phase 1 Simplex iterations
    while np.any(tableau[-1, :-1] > 0):
        pivot_col = np.argmax(tableau[-1, :-1])
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf
        pivot_row = np.argmin(ratios)

        if ratios[pivot_row] == np.inf:
            raise Exception("Unbounded solution")

        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        basis_vars[pivot_row] = pivot_col + 1  # Update basis variable

        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        print(f"\nTableau after Pivot (Row: {pivot_row+1}, Col: {pivot_col+1}):")
        print(format_tableau(tableau, basis_vars))

    if tableau[-1, -1] > 0:
        raise Exception("Infeasible Solution")

    print("\nFinal Tableau - Phase 1 (Proceeding to Phase 2):")
    print(format_tableau(tableau, basis_vars))

    # Phase 2: Remove artificial variables
    tableau = tableau[:, :n+1]  # Remove artificial variable columns
    tableau[-1, :-1] = c
    tableau[-1, -1] = 0

    print("\nInitial Tableau - Phase 2:")
    print(format_tableau(tableau, basis_vars))

    # Phase 2 Simplex iterations
    while np.any(tableau[-1, :-1] > 0):
        pivot_col = np.argmax(tableau[-1, :-1])
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf
        pivot_row = np.argmin(ratios)

        if ratios[pivot_row] == np.inf:
            raise Exception("Unbounded solution")

        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        basis_vars[pivot_row] = pivot_col + 1  # Update basis variable

        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        print(f"\nTableau after Pivot (Row: {pivot_row+1}, Col: {pivot_col+1}):")
        print(format_tableau(tableau, basis_vars))

    print("\nFinal Tableau - Phase 2 (Optimal Solution Found):")
    print(format_tableau(tableau, basis_vars))
    return tableau[-1, -1]
