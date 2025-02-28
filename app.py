import streamlit as st
import numpy as np
import pandas as pd
from two_phase_simplex import two_phase_simplex, format_tableau

st.title("Two-Phase Simplex Method Solver")

m = st.number_input("Number of Constraints (m)", min_value=1, step=1)
n = st.number_input("Number of Variables (n)", min_value=1, step=1)

if st.button("Proceed"):
    A = []
    b = []
    c = []

    st.subheader("Enter Constraint Coefficients")
    for i in range(int(m)):
        row = st.text_input(f"Constraint {i+1} (Space Separated)", "")
        if row:
            A.append(list(map(float, row.split())))
            b.append(float(st.text_input(f"Right-hand side of Constraint {i+1}", "")))

    st.subheader("Enter Objective Function Coefficients")
    obj = st.text_input("Objective Function Coefficients (Space Separated)", "")
    if obj:
        c = list(map(float, obj.split()))

    if len(A) == m and len(c) == n:
        A = np.array(A)
        b = np.array(b)
        c = np.array(c)

        st.subheader("Solving Two-Phase Simplex")
        try:
            result = two_phase_simplex(A, b, c)
            st.success(f"Optimal Solution: {result}")
        except Exception as e:
            st.error(str(e))
