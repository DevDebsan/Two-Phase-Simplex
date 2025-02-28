import streamlit as st

def print_tableau(tableau, phase):
    st.write(f"### Phase {phase} Tableau")
    for row in tableau:
        st.write("\t".join(map(lambda x: f"{x:.2f}", row)))

def find_pivot_column(tableau):
    last_row = tableau[-1]
    min_value = min(last_row[:-1])
    if min_value >= 0:
        return None  # Optimal solution reached
    return last_row.index(min_value)

def find_pivot_row(tableau, pivot_col):
    ratios = []
    for i in range(len(tableau) - 1):
        if tableau[i][pivot_col] > 0:
            ratios.append((tableau[i][-1] / tableau[i][pivot_col], i))
    if not ratios:
        return None  # Unbounded
    return min(ratios)[1]

def perform_pivot(tableau, pivot_row, pivot_col):
    pivot_element = tableau[pivot_row][pivot_col]
    tableau[pivot_row] = [x / pivot_element for x in tableau[pivot_row]]
    for i in range(len(tableau)):
        if i != pivot_row:
            factor = tableau[i][pivot_col]
            tableau[i] = [tableau[i][j] - factor * tableau[pivot_row][j] for j in range(len(tableau[i]))]

def simplex(tableau, phase):
    while True:
        print_tableau(tableau, phase)
        pivot_col = find_pivot_column(tableau)
        if pivot_col is None:
            break
        pivot_row = find_pivot_row(tableau, pivot_col)
        if pivot_row is None:
            st.write("Problem is unbounded.")
            return None
        perform_pivot(tableau, pivot_row, pivot_col)
    return tableau

def two_phase_simplex():
    st.title("Two-Phase Simplex Method")
    num_vars = st.number_input("Enter number of variables:", min_value=1, step=1, value=2)
    num_constraints = st.number_input("Enter number of constraints:", min_value=1, step=1, value=2)
    
    tableau = []
    for i in range(num_constraints + 1):
        row = st.text_input(f"Enter row {i+1} coefficients (space-separated):")
        if row:
            tableau.append(list(map(float, row.split())))
    
    if st.button("Solve LP"):
        st.write("### Phase 1")
        tableau = simplex(tableau, 1)
        
        if tableau is not None:
            st.write("### Phase 2")
            tableau = simplex(tableau, 2)
            st.write("Optimal solution found!")

two_phase_simplex()
