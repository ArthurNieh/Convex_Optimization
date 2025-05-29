
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

inner_value_s = []

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
    # if not np.allclose(get_gradient_value(x, a, b, c, t), g):
    #     print("Gradient is not equal.")
    #     print("g:", g)
    #     print("Calculated gradient:", get_gradient_value(x, a, b, c, t))
    #     exit(0)
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
        phi -= np.log(s_i)

    return value * t + phi

def get_gradient_value(x, a, b, c, t):
    m = a.shape[0]
    g = np.zeros(m)
    sigmoid = 1 / (1 + np.exp(-a @ x))
    for i in range(m):
        g[i] = a[i] * sigmoid + x[i]
    
    g_phi = np.zeros_like(x, dtype=float)
    for i in range(b.shape[0]):
        s_i = c[i] - b[i] @ x
        g_phi += b[i] / s_i
    
    return g*t + g_phi

def newton_method(a, b, c, t, x0):
    x = x0.copy()
    alpha = 0.25
    beta = 0.5
    inner_iter = 0
    while True:
        NT_step, lamda_2 = newton_step(x, a, b, c, t)

        if lamda_2/2 <= 1e-10:
            break
        
        t0 = 1
        while True:
            x_new = x + NT_step * t0
            if np.any(c - np.dot(b, x_new) <= 0):
                # print("Constraint violation: c - b @ x must be positive.")
                t0 *= beta
                continue

            new_value = get_function_value(x_new, a, t, b, c)
            gradient = get_gradient_value(x, a, b, c, t)
            threshold = get_function_value(x, a, t, b, c) + np.dot(alpha * t0 * gradient.transpose(), NT_step)
            if new_value <= threshold:
                break
            t0 = beta * t0

        x = x_new
        inner_iter += 1
        inner_value_s.append(get_function_value(x, a, 0, b, c))
    return x, inner_iter

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
    return problem.value, x.value


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
    inner_iter_s = []
    fi_s = []
    lamda_i_s = []

    center_step = 0
    for i in range(maxiter := 100):
        x, inner_iter = newton_method(a, b, c, t, x)
        center_step += 1
        # print(f"Current x: {x}")
        value = get_function_value(x, a, 0, b, c)
        fi = np.dot(b, x) - c
        lamda_i = -1/t/fi

        value_s.append(value)
        t_s.append(t)
        glamda_s.append(value - m/t)
        inner_iter_s.append(inner_iter)
        fi_s.append(fi)
        lamda_i_s.append(lamda_i)
        # print(f"Iteration {i}\nt={t}: value={value}, glamda={value - m/t}")

        if m/t < 1e-6:
            break
        t *= u
    
    print(f"Optimal x: {x}")
    print(f"Optimal value: {value}")
    print(f"Inner iterations per step: {sum(inner_iter_s)}")
    print(f"Total center steps: {center_step}")

    # Plotting the results
    
    print(f"final slack variables (c - b @ x): {fi_s[-1]}")
    print(f"Dual variables (lamda_i): {lamda_i_s[-1]}")
    
    plt.title('dual value vs. t')
    plt.ylabel('lamda_i')
    plt.xlabel('t')
    plt.xscale('log')
    plt.grid(True)
    plt.plot(t_s, [lam[0] for lam in lamda_i_s], 'b-o')
    plt.plot(t_s, [lam[1] for lam in lamda_i_s], 'r-o')
    plt.plot(t_s, [lam[2] for lam in lamda_i_s], 'g-o')
    plt.legend(['lambda[1] (blue)', 'lambda[2] (red)', 'lambda[3] (green)'])
    plt.savefig('lamda_i.png')
    plt.show()

    plt.title('Constraint value vs. t')
    plt.ylabel('f_i')
    plt.xlabel('t')
    plt.xscale('log')
    plt.grid(True)
    plt.plot(t_s, [fi[0] for fi in fi_s], 'b-o')
    plt.plot(t_s, [fi[1] for fi in fi_s], 'r-o')
    plt.plot(t_s, [fi[2] for fi in fi_s], 'g-o')
    plt.legend(['f[1] (blue)', 'f[2] (red)', 'f[3] (green)'])
    plt.savefig('f_i.png')
    plt.show()


    plt.title('Primal and Dual values vs. t')
    plt.ylabel('value')
    plt.xlabel('t')
    plt.xscale('log')
    plt.grid(True)
    plt.plot(t_s, value_s, 'b-o')
    plt.plot(t_s, glamda_s, 'r-o')
    plt.legend(['f (blue)', 'g (red)'])
    plt.savefig('primal_dual.png')
    plt.show()

    plt.title('Duality gap vs. t')
    plt.ylabel('value')
    plt.xlabel('t')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.plot(t_s, [value_s[i] - glamda_s[i] for i in range(len(value_s))], 'b-o')
    plt.savefig('duality_gap.png')
    plt.show()

    plt.title('Objective value vs. inner iteration')
    plt.ylabel('objective value')
    plt.xlabel('inner iteration')
    plt.plot(inner_value_s, 'b-o')
    plt.savefig('objective_value_inner.png')
    plt.show()

    cvxpy_value, cvxpy_x = solve_by_cvxpy(a, b, c)
    print("Difference in optimal value:", abs(value - cvxpy_value))
    print("Difference in optimal x:", np.linalg.norm(x - cvxpy_x))