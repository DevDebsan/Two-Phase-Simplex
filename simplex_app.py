import streamlit as st

class SimplexSolver:
    def __init__(self):
        self.max_iter = 50
        self.iobj = None
        self.irows = []
        self.variables = []
        self.pivots = []
        self.target = None
        self.r_vector = None
        self.matrix_a = None
        self.cost_vector = []
        self.p1_cost_vector = None
        self.basic_vars = None
        self.basis = None
        self.c_bfs = None
        self.dim = None
        self.r_cost = None
        self.minmax_r_cost = None
        self.minmax_r_cost_index = None
        self.ratio = None
        self.leaving_index = None
        self.kount = 1
        self.obj_z = None
        self.basic_kount = 0
        self.non_basic_kount = 0
        self.artificial_kount = 0
        self.unbounded = False
        self.history = []

    def find_terms(self, row):
        terms = row.split()
        return [term.strip() for term in terms if term.strip()]

    def find_coeff(self, row):
        vars = {}
        for term in row:
            variable = ''.join(filter(str.isalpha, term))
            i = term.find(variable)
            value = term[:i].strip()
            coeff = 1
            if value == '' or value == '+':
                coeff = 1
            elif value == '-':
                coeff = -1
            else:
                try:
                    coeff = float(value)
                except ValueError:
                    raise ValueError(f"Invalid coefficient in term: {term}")
            if variable not in self.variables:
                self.variables.append(variable)
            vars[variable] = coeff
        return vars

    def parse_obj(self, iobj):
        try:
            mtarget, row = iobj.split('=')
            target = mtarget.strip().lower()
            obj_value = self.find_coeff(self.find_terms(row))
            return target, obj_value
        except ValueError as e:
            raise ValueError(f"Invalid objective function format: {e}")

    def parse_constraint(self, irows):
        signs = []
        rows = []
        for row in irows:
            if not row.strip():
                continue
            le = row.split('<=')
            if len(le) == 2:
                signs.append('le')
                rows.append(le)
                continue
            ge = row.split('>=')
            if len(ge) == 2:
                signs.append('ge')
                rows.append(ge)
                continue
            eq = row.split('=')
            if len(eq) == 2:
                signs.append('e')
                rows.append(eq)
                continue
            raise ValueError(f"Invalid constraint format: {row}")
        
        r_vector = []
        coeff_dict = []
        for row in rows:
            try:
                r_vector.append(float(row[1].strip()))
                row_terms = self.find_terms(row[0])
                coeff_dict.append(self.find_coeff(row_terms))
            except (IndexError, ValueError) as e:
                raise ValueError(f"Error parsing constraint: {row}. Details: {e}")
        
        return r_vector, coeff_dict, signs

    def get_cost_vector(self, obj):
        self.cost_vector = []
        for v in self.variables:
            if v in obj:
                self.cost_vector.append(obj[v])
            else:
                self.cost_vector.append(0)

    def find_b_negative(self, b):
        return [i for i, v in enumerate(b) if v < 0]

    def remove_b_negative(self, b_index, c_dict, r_vector):
        for i in b_index:
            for k in c_dict[i]:
                c_dict[i][k] *= -1
            r_vector[i] *= -1

    def assign_zero_coeff(self, c_dict):
        for row in c_dict:
            for v in self.variables:
                if v not in row:
                    row[v] = 0

    def form_matrix_a(self, c_dict):
        return [[row.get(v, 0) for v in self.variables] for row in c_dict]

    def find_remaining(self, matrix, i):
        return matrix[:i] + matrix[i+1:]

    def add_vars(self, q, i):
        row_with_1 = self.matrix_a[i] + [-1 if q == 'srpls' else 1]
        self.variables.append(f'{q}{i}')
        remaining_rows = self.find_remaining(self.matrix_a, i)
        new_remaining_rows = [row + [0] for row in remaining_rows]
        new_remaining_rows.insert(i, row_with_1)
        self.matrix_a = new_remaining_rows
        if q != 'artfcl':
            self.cost_vector.append(0)
        return len(row_with_1) - 1

    def add_slack_surplus_artificial(self, signs):
        for i, sign in enumerate(signs):
            if sign == 'le':
                pivot = self.add_vars('slck', i)
                self.pivots.append(pivot)
            elif sign == 'ge':
                self.add_vars('srpls', i)
                pivot = self.add_vars('artfcl', i)
                self.pivots.append(pivot)

    def standard_form(self, iobj, irows):
        target, obj_value = self.parse_obj(iobj)
        r_vector, coeff_dict, signs = self.parse_constraint(irows)
        self.get_cost_vector(obj_value)
        b_negative_index = self.find_b_negative(r_vector)
        self.remove_b_negative(b_negative_index, coeff_dict, r_vector)
        self.assign_zero_coeff(coeff_dict)
        self.matrix_a = self.form_matrix_a(coeff_dict)
        self.basic_kount = len(self.variables)
        self.add_slack_surplus_artificial(signs)
        self.non_basic_kount = len(self.variables) - self.basic_kount
        self.artificial_kount = len(self.variables) - (self.basic_kount + self.non_basic_kount)
        return target, r_vector

    def get_phase1_cost_vector(self):
        return [1 if 'artfcl' in v else 0 for v in self.variables]

    def get_bfs(self):
        arr = [0] * len(self.variables)
        for i, p in enumerate(self.pivots):
            arr[p] = self.r_vector[i]
        return arr

    def dot_p(self, v1, v2):
        return sum(q1 * q2 for q1, q2 in zip(v1, v2))

    def v_divide(self, v1, v2):
        return [q1 / q2 for q1, q2 in zip(v1, v2)]

    def v_subtract(self, v1, v2):
        return [q1 - q2 for q1, q2 in zip(v1, v2)]

    def get_cj_bar(self, col, c_vector, basis):
        p = [self.matrix_a[i][col] for i in range(self.dim[0])]
        return c_vector[col] - self.dot_p(p, basis)

    def find_r_cost(self, c_vector):
        return [self.get_cj_bar(j, c_vector, self.basis) for j in range(self.dim[1])]

    def get_basic_vars(self):
        return [self.variables[p] for p in self.pivots]

    def get_basis(self, c_vector):
        return [c_vector[p] for p in self.pivots]

    def get_dim(self):
        return [len(self.r_vector), len(self.variables)]

    def find_leaving_var(self, col):
        p = [self.matrix_a[i][col] for i in range(self.dim[0])]
        self.ratio = self.v_divide(self.r_vector, p)
        filtered_ratio = [q for q in self.ratio if q >= 0 and q != float('inf')]
        if not filtered_ratio:
            self.unbounded = True
            return -1
        min_ratio = min(filtered_ratio)
        return self.ratio.index(min_ratio)

    def row_operation(self, row, col):
        element = self.matrix_a[row][col]
        self.matrix_a[row] = [q / element for q in self.matrix_a[row]]
        self.r_vector[row] /= element
        remaining_rows = self.find_remaining(self.matrix_a, row)
        r_remaining = self.find_remaining(self.r_vector, row)
        pivot_row = self.matrix_a[row]
        r_pivot = self.r_vector[row]

        self.matrix_a = [self.v_subtract(r, [q * r[col] for q in pivot_row]) for r in remaining_rows]
        self.matrix_a.insert(row, pivot_row)
        self.r_vector = r_remaining
        self.r_vector.insert(row, r_pivot)

    def update_pivot(self, row, col):
        self.pivots[row] = col

    def contains_artificial(self):
        return any('artfcl' in b for b in self.basic_vars)

    def find_target_r_cost(self, target, r_cost):
        if target == 'min':
            min_r_cost = min(r_cost)
            return min_r_cost if min_r_cost < 0 else None
        max_r_cost = max(r_cost)
        return max_r_cost if max_r_cost > 0 else None

    def get_soln(self, v):
        return self.dot_p(v, self.c_bfs)

    def check_history(self):
        s = f'{self.minmax_r_cost_index}{self.leaving_index}{self.obj_z}'
        if s in self.history:
            return False
        self.history.append(s)
        return True

    def check_decimals(self, n):
        s = str(n)
        if '.' in s and len(s.split('.')[1]) > 5:
            return round(n, 5)
        return n

    def simplex(self, phase):
        self.basis = self.get_basis(self.p1_cost_vector if phase == 1 else self.cost_vector)
        self.c_bfs = self.get_bfs()
        self.obj_z = self.get_soln(self.p1_cost_vector if phase == 1 else self.cost_vector)
        self.r_cost = self.find_r_cost(self.p1_cost_vector if phase == 1 else self.cost_vector)
        self.minmax_r_cost = min(self.r_cost) if phase == 1 else self.find_target_r_cost(self.target, self.r_cost)
        if not self.minmax_r_cost:
            return False
        self.minmax_r_cost_index = self.r_cost.index(self.minmax_r_cost)
        self.leaving_index = self.find_leaving_var(self.minmax_r_cost_index)
        if self.leaving_index == -1:
            return False
        if not self.check_history():
            return False
        self.row_operation(self.leaving_index, self.minmax_r_cost_index)
        self.update_pivot(self.leaving_index, self.minmax_r_cost_index)
        return True

    def remove_artificial(self):
        artificial_index = [i for i, v in enumerate(self.variables) if 'artfcl' in v]
        self.variables = [v for i, v in enumerate(self.variables) if i not in artificial_index]
        self.pivots = [p - sum(1 for ai in artificial_index if ai < p) for p in self.pivots]
        self.c_bfs = [q for i, q in enumerate(self.c_bfs) if i not in artificial_index]
        self.matrix_a = [[q for i, q in enumerate(row) if i not in artificial_index] for row in self.matrix_a]

    def phase1(self):
        self.dim = self.get_dim()
        self.p1_cost_vector = self.get_phase1_cost_vector()
        while self.kount <= self.max_iter:
            self.basic_vars = self.get_basic_vars()
            if not self.contains_artificial():
                break
            if not self.simplex(1):
                break
            self.kount += 1
        if self.kount == self.max_iter + 1:
            return
        if self.unbounded:
            return
        self.remove_artificial()

    def phase2(self):
        self.dim = self.get_dim()
        while self.kount <= self.max_iter:
            self.basic_vars = self.get_basic_vars()
            if not self.simplex(2):
                break
            self.kount += 1
        if self.kount == self.max_iter + 1:
            return

    def start_simplex(self):
        self.basic_vars = self.get_basic_vars()
        if self.contains_artificial():
            self.phase1()
        if not self.unbounded:
            self.phase2()

    def get_output(self):
        return {
            'objective_value': self.obj_z,
            'basic_variables': self.basic_vars,
            'non_basic_variables': [v for v in self.variables if v not in self.basic_vars],
            'solution_vector': self.c_bfs
        }

    def get_steps(self):
        return self.history

def main():
    st.set_page_config(
        page_title="Two-Phase Simplex Method",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.example.com',
            'Report a bug': "https://www.example.com",
            'About': "This is a Two-Phase Simplex Method solver."
        }
    )

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #262730;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Two-Phase Simplex Method Calculator")
    st.write("Enter the linear programming problem below:")

    iobj = st.text_input("Objective Function (e.g., max = 3x1 + 2x2 + 3x3):")
    irows = st.text_area("Constraints (one per line, e.g., 2x1 + x2 <= 18):").split('\n')

    if st.button("Solve"):
        if not iobj or not irows:
            st.error("Please provide both the objective function and constraints.")
        else:
            solver = SimplexSolver()
            try:
                target, r_vector = solver.standard_form(iobj, irows)
                solver.target = target
                solver.r_vector = r_vector
                solver.start_simplex()

                st.subheader("Solution:")
                st.write(solver.get_output())

                st.subheader("Steps:")
                st.write(solver.get_steps())
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
