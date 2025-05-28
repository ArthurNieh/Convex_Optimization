
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def newton_step(x, a, b, c, t):
    # returns the Newton step ∆xnt and the Newton decrement
    # ∆xnt = -H(x)^{-1} g(x)
    # lamda^2 = -∆xnt^T g ∆xnt
    # g(x) = ∇f(x)
    # H(x) = ∇²f(x) = Hessian of f at x
    
    m = a.shape[0]
    g = np.zeros(m)
    H = np.zeros((m, m))

    sigmoid = 1 / (1 + np.exp(-a @ x))

    for i in range(m):
        g[i] = a[i] * sigmoid + x[i]
        for j in range(m):
            if i == j:
                H[i, j] = (a[i]**2) * sigmoid * (1-sigmoid) + 1
            else:
                H[i, j] = a[i]*a[j] * sigmoid * (1-sigmoid)
    
    g_phi = np.zeros_like(x, dtype=float)
    for i in range(b.shape[0]):
        s_i = c[i] - b[i] @ x
        g_phi += b[i] / s_i

    h_phi = np.zeros((x.size, x.size))
    for i in range(b.shape[0]):
        s_i = c[i] - b[i] @ x
        bi = b[i].reshape(-1, 1)  # Column vector
        h_phi += (bi @ bi.T) / (s_i ** 2)

    g = g*t + g_phi
    H = H*t + h_phi

    H_inv = np.linalg.inv(H)
    NT_step = - np.dot(H_inv, g)

    NT_step_trans = NT_step.transpose()
    lamda_2 = np.dot(np.dot(NT_step_trans, H), NT_step)

    return NT_step, lamda_2

def get_function_value(x, a, t, b, c):
    value = np.log(1 + np.exp(a @ x)) + 1/2*(sum(i**2 for i in x))

    if t == 0:
        return value

    phi = 0
    for i in range(b.shape[0]):
        s_i = c[i] - b[i] @ x
        if s_i <= 0:
            print(c[i], b[i], x)
            raise ValueError("Constraint violation: s_i must be positive.")
        phi += np.log(s_i)

    return value * t - phi

def get_gradient_value(x, a, b, c):
    m = a.shape[0]
    g = np.zeros(m)
    sigmoid = 1 / (1 + np.exp(-a @ x))
    for i in range(m):
        g[i] = a[i] * sigmoid + x[i]
    
    g_phi = np.zeros_like(x, dtype=float)
    for i in range(b.shape[0]):
        s_i = c[i] - b[i] @ x
        g_phi += b[i] / s_i
    
    return g + g_phi

def newton_method(a, b, c, t, x0):
    x = x0.copy()
    alpha = 0.25
    beta = 0.5
    while True:
        NT_step, lamda_2 = newton_step(x, a, b, c, t)

        if lamda_2/2 <= 1e-10:
            break
        
        t0 = 1
        while True:
            new_value = get_function_value(x+t*NT_step, a, t, b, c)
            gradient = get_gradient_value(x, a, b, c)
            threshold = get_function_value(x, a, t, b, c) + np.dot(alpha * t0 * gradient.transpose(), NT_step)
            if new_value <= threshold:
                break
            t0 = beta * t0

        x = x + NT_step * t0
    return x

def solve_by_cvxpy(a, b, c):
    x = cp.Variable(5)
    z = a @ x
    
    objective = cp.Minimize(
        cp.log_sum_exp(cp.hstack([0, z])) + 0.5 * cp.sum_squares(x)
    )

    constraints = [b @ x <= c]

    problem = cp.Problem(objective, constraints)
    problem.solve()
    print("cvxpy Optimal value:", problem.value)
    print("cvxpy Optimal x:", x.value)


if __name__ == "__main__":

    b = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, -1, 0],
                  [0, 0, 1, 0, -1]])
    c = np.array([3, 2, 0.1])

    t = 1
    u = 10
    m = 3
    x = np.array([0, 0, 0, 0, 0])
    a = np.array([1, 2, 3, 4, 5])

    value_s = []
    t_s = []
    glamda_s = []

    # while True:
    #     x = newton_method(a, b, c, t, x)
    #     value = get_function_value(x, a, 0, b, c)

    #     value_s.append(value)
    #     t_s.append(t)
    #     glamda_s.append(value - m/t)

    #     if m/t < 1e-6:
    #         break
    #     t *= u
    
    # print(f"Optimal x: {x}")
    # print(f"Optimal value: {value}")

    solve_by_cvxpy(a, b, c)