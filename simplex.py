"""
This module applies the simplex algorithm for LP problems.
It can handle minimization and maximization problems.
It can handle upper bounds, lower bounds and equality constraints.
It can handle negative RHS values in constraints.
It uses Bland's rule for entering and exiting variables to prevent cycling.
It assumes all variables are restricted to be >= 0
To handle artificial variables, the big M method is used.
To represent M variables I chose to use complex numbers, where "i" is treated as M,
    this doesn't work for all computations, but in this problem it doesn't cause any issues.
"""
import argparse
import time

import numpy as np

from utils import print_problem, print_solution, read_input_file


def get_entering_var(Cb_Binv_A, C):
    """Get the entering variable index using Bland's rule."""
    optimality_vec = Cb_Binv_A - C
    entering_ind = None
    for i, elt in enumerate(optimality_vec):
        # print("i, ", i, ", elt, ", elt)
        if elt.imag < -1e-9 or (abs(elt.imag) <= 1e-9 and elt.real < -1e-9):
            entering_ind = i
            break
    return entering_ind


def get_exiting_var(Binv_A, Binv_b, entering_ind):
    """
    Get the exiting variable index using min-ratio test.
    In case of a tie, we use Bland's rule to select a variable.
    """
    min_ratio = float("inf")
    min_ind = None
    for i, rhs_val in enumerate(Binv_b):
        col_val = Binv_A[i, entering_ind]
        if col_val > 1e-9:
            ratio = rhs_val / col_val
            if ratio < min_ratio:
                min_ratio = ratio
                min_ind = i
    return min_ind


def check_feasible(X, C):
    """Check if the current basis contains any artificial variables."""
    is_feasible = True
    for ind in X:
        if abs(C[ind].imag) >= 1e-9:
            is_feasible = False
            break
    return is_feasible


def simplex_step(A, C, b, X):
    """Apply one iteration of the simplex method."""
    is_opt = False
    is_unbounded = False
    is_infeasible = False
    B = A[:, X]
    Cb = C[X]
    Binv = np.linalg.inv(B)
    Binv_A = np.matmul(Binv, A)
    Binv_b = np.matmul(Binv, b)
    Cb_Binv_A = np.matmul(Cb, Binv_A)
    obj_val = np.matmul(Cb, Binv_b)
    print_solution(X, Binv_b, obj_val)

    entering_ind = get_entering_var(Cb_Binv_A, C)
    # If there is no entering variable, we are optimal.
    # But we still need to check if all artificial variables are removed from the basis
    if entering_ind is None:
        is_infeasible = not check_feasible(X, C)
        if not is_infeasible:
            is_opt = True
            obj_val = np.matmul(Cb, Binv_b)
    # If there is an entering variable, find the exiting variable and update basis
    if entering_ind is not None:
        exiting_ind = get_exiting_var(Binv_A, Binv_b, entering_ind)
        print(f"Entering variable is X{entering_ind}")
        # If there is no exiting variable found, the problem is unbounded
        if exiting_ind is None:
            is_unbounded = True
        else:
            print(f"Exiting variable is X{X[exiting_ind]}")
            X[exiting_ind] = entering_ind
    return X, is_opt, is_unbounded, is_infeasible


def simplex(A, C, b, X):
    counter = 0
    while True:
        X, is_opt, is_unbounded, is_infeasible = simplex_step(A, C, b, X)
        if any((is_opt, is_unbounded, is_infeasible)):
            break
        print("-" * 80)
        counter += 1
        time.sleep(0.5)
    if is_opt:
        print("-" * 80)
        print(f"Optimal solution found in {counter} steps.")
    elif is_unbounded:
        print("The problem is unbounded!")
    else:
        print("No feasible solutions exist for this problem.")


def make_canonical(A, C, b, constraint_types, is_max):
    """
    Converts the problem into a maximization problem with positive right hand size.
    Converts all the constraints into equality constraints, adding slack, surplus and
    artificial variables as necessary. This function also returns our initial basis.
    """
    # Convert the problem into a maximization problem if necessary
    if not is_max:
        C *= -1
    num_eqs = A.shape[0]
    basis_inds = np.empty((num_eqs), dtype=np.uint64)
    for i, c_type in enumerate(constraint_types):
        # If there is a negative RHS value, multiply everything by -1
        if b[i] < 0.0:
            A[i, :] *= -1
            b[i] *= -1
            if constraint_types[i] == ">=":
                c_type = "<="
            elif constraint_types[i] == "<=":
                c_type = ">="
        # Handle lesser than inequality by adding a slack variable
        if c_type == "<=":
            new_acol = np.zeros((num_eqs, 1))
            new_acol[i] = 1.0
            new_c_elts = np.array([0.0], dtype=complex)
        # Handle greater than inequality by adding surplus and artificial variables
        elif c_type == ">=":
            new_acol = np.zeros((num_eqs, 2))
            # Surplus variable
            new_acol[i, 0] = -1.0
            # Artificial variable
            new_acol[i, 1] = 1.0
            new_c_elts = np.array([0.0, -1j], dtype=complex)
        # Handle equality constraints by adding an artificial variable
        else:
            new_acol = np.zeros((num_eqs, 1))
            new_acol[i] = 1.0
            new_c_elts = np.array([-1j], dtype=complex)
        A = np.concatenate((A, new_acol), axis=1)
        C = np.concatenate((C, new_c_elts), axis=0)
        basis_inds[i] = A.shape[1] - 1
    return A, C, b, basis_inds


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Runs the the simplex algorithm.")
    parser.add_argument("-i", "--input", default="input.txt", type=str,
                        help="File containing input information.")
    parser.add_argument("--min", action="store_true", help="Minimize the objective function. If \
                        this argument is not used, maximization problem is assumed.")
    return parser.parse_args()


def main():
    args = parse_args()
    A, C, b, constraint_types = read_input_file(args.input)
    print(f"### Reading problem from input file {args.input}.")
    print("-" * 80)
    print_problem(A, C, b, constraint_types, not args.min)
    A, C, b, basis_inds = make_canonical(A, C, b, constraint_types, not args.min)
    print("-" * 80)
    print("### Succesfully changed the problem to canonical form.")
    print_problem(A, C, b, ["="] * A.shape[0], True)
    print("-" * 80)
    simplex(A, C, b, basis_inds)


if __name__ == "__main__":
    main()
