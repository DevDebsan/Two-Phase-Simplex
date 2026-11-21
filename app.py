import contextlib
import io

import numpy as np
import streamlit as st

from simplex_app import two_phase_simplex

st.title("Two-Phase Simplex Method Solver")
st.caption("Minimise c·x subject to A·x = b, x ≥ 0. Enter numbers separated by spaces.")

m = int(st.number_input("Number of constraints (m)", min_value=1, value=2, step=1))
n = int(st.number_input("Number of variables (n)", min_value=1, value=2, step=1))

with st.form("lp"):
    st.subheader("Constraints  (A x = b)")
    rows = [st.text_input(f"Coefficients of constraint {i + 1}  ({n} values)", key=f"a{i}") for i in range(m)]
    rhs = [st.text_input(f"Right-hand side of constraint {i + 1}", key=f"b{i}") for i in range(m)]

    st.subheader("Objective  (minimise c x)")
    obj = st.text_input(f"Objective coefficients  ({n} values)")
    submitted = st.form_submit_button("Solve")

if submitted:
    try:
        A = np.array([list(map(float, r.split())) for r in rows])
        b = np.array([float(v) for v in rhs])
        c = np.array(list(map(float, obj.split())))
        if A.shape != (m, n) or c.shape != (n,):
            raise ValueError(f"Expected {m} rows of {n} coefficients and {n} objective coefficients.")

        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            value = two_phase_simplex(A, b, c)
        st.success(f"Optimal value: {value:.6g}")
        with st.expander("Show tableaux"):
            st.code(log.getvalue())
    except Exception as e:  # infeasible, unbounded, or bad input
        st.error(str(e))
