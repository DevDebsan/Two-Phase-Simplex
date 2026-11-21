# Two-Phase Simplex Solver

A Python implementation of the **Two-Phase Simplex method** for linear programming, with a small
[Streamlit](https://streamlit.io) interface that shows every tableau along the way.

Solves: **minimise** `c·x` subject to `A·x = b`, `x ≥ 0`.

## How it works
1. **Phase 1** adds artificial variables and minimises their sum to find a feasible starting point
   (or reports the problem as **infeasible**).
2. Leftover artificial variables are pivoted out of the basis; redundant constraints are dropped.
3. **Phase 2** restores the real objective and iterates to the optimum (or reports **unbounded**).

Bland's rule is used for pivoting, so degenerate problems terminate.

## Run it
```bash
pip install streamlit numpy pandas
streamlit run app.py
```

Or use the solver directly:
```python
import numpy as np
from simplex_app import two_phase_simplex

# minimise 2x1 + 3x2  s.t.  x1 + x2 = 4,  x1 + 3x2 = 6
two_phase_simplex(np.array([[1, 1], [1, 3]]), np.array([4, 6]), np.array([2, 3]))  # -> 9.0
```

## Testing
Results were checked against `scipy.optimize.linprog` on 300 random problems, covering optimal,
infeasible and unbounded cases.

## Files
- `simplex_app.py`: the solver (`two_phase_simplex`, `format_tableau`)
- `app.py`: Streamlit front end

**Stack:** Python · NumPy · pandas · Streamlit
