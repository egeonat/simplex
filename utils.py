import numpy as np


def print_problem(A, C, b, constraint_types, is_max):
    print("### Our problem has the form:")
    if is_max:
        problem_type = "Maximize"
    else:
        problem_type = "Minimize"
    print(f"{problem_type} objective function:")
    obj_str = "Z ="
    for i, coef in enumerate(C):
        obj_str += " "
        if coef.imag == 0:
            if coef >= 0 and i > 0:
                obj_str += "+"
            obj_str += f"{coef.real}"
        else:
            if coef.real == 0:
                if coef.imag > 0 and i > 0:
                    obj_str += "+"
                obj_str += f"{coef.imag}*M"
            else:
                if i > 0:
                    obj_str += "+"
                obj_str += f"({coef.real} "
                if coef.imag >= 0.0:
                    obj_str += "+"
                obj_str += f"{coef.imag}*M)"
        obj_str += f"*X{i}"
    print(obj_str)

    print("Subject to constraints:")
    for i, row in enumerate(A):
        eq_str = ""
        for j, coef in enumerate(row):
            if coef >= 0 and j > 0:
                eq_str += "+"
            eq_str += f"{coef}*X{j} "
        eq_str += f"{constraint_types[i]} {b[i]}"
        print(eq_str)


def print_solution(X, rhs, obj_val):
    print("Basis variables and their values are:")
    for i, ind in enumerate(X):
        print(f"X{ind} = {rhs[i]}")
    if obj_val.imag == 0:
        obj_str = f"{obj_val.real}"
    elif obj_val.real == 0:
        obj_str = f"{obj_val.imag}M"
    else:
        obj_str = f"({obj_val.real}"
        if obj_val.imag >= 0:
            obj_str += " + "
        obj_str += f"{obj_val.imag}M)"
    print(f"Objective function value is: {obj_str}")


def read_input_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    ind = 0
    while lines[ind][0] == "#" or lines[ind][0].isspace():
        ind += 1
    C = np.array([float(num) for num in lines[ind].split()], dtype=complex)
    ind += 1
    while lines[ind][0] == "#" or lines[ind][0].isspace():
        ind += 1
    constraints = [l_str.split() for l_str in lines[ind:]]
    A = np.array([[float(num) for num in equation[:-2]] for equation in constraints])
    b = np.array([float(equation[-1]) for equation in constraints])
    constraint_types = [equation[-2] for equation in constraints]
    assert all([(c in ["=", "<=", ">="]) for c in constraint_types])
    return A, C, b, constraint_types
