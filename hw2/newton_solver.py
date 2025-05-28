

# Newton's method for solving nonlinear equations
# f(x) = \log(1 + \exp(a^T x)) + \frac{1}{2} \sum_{i=1}^5 x_i^2
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def newton_step(x, a):
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
    
    H_inv = np.linalg.inv(H)
    NT_step = - np.dot(H_inv, g)

    NT_step_trans = NT_step.transpose()
    lamda_2 = np.dot(np.dot(NT_step_trans, H), NT_step)

    return NT_step, lamda_2

def get_function_value(x, a):
    value = np.log(1 + np.exp(a @ x)) + 1/2*(sum(i**2 for i in x))
    return value

def get_gradient_value(x, a):
    m = a.shape[0]
    g = np.zeros(m)
    sigmoid = 1 / (1 + np.exp(-a @ x))
    for i in range(m):
        g[i] = a[i] * sigmoid + x[i]
    return g

def solve_by_cvxpy(a):
    x = cp.Variable(5)
    z = a @ x
    
    objective = cp.Minimize(
        cp.log_sum_exp(cp.hstack([0, z])) + 0.5 * cp.sum_squares(x)
    )
    problem = cp.Problem(objective)
    problem.solve()
    print("cvxpy Optimal value:", problem.value)
    print("cvxpy Optimal x:", x.value)

if __name__ == "__main__":
    alpha = 0.25
    beta = 0.5
    epsilon = (10**(-10))

    # starting point = (0, 0, 0, 0, 0)
    x = np.array([0, 0, 0, 0, 0])
    a = np.array([1, 2, 3, 4, 5])

    objective = []
    NT_decrement = []
    i = 0
    while True:
        NT_step, lamda_2 = newton_step(x, a)
        
        objective.append(get_function_value(x, a))
        NT_decrement.append(lamda_2/2)

        if lamda_2/2 <= epsilon:
            break
        
        t = 1
        while True:
            new_value = get_function_value(x+t*NT_step, a)
            gradient = get_gradient_value(x, a)
            threshold = get_function_value(x, a) + np.dot(alpha * t * gradient.transpose(), NT_step)
            if new_value <= threshold:
                break
            t = beta * t

        x = x + NT_step * t

    print("Optimal value:", get_function_value(x,a))
    print("Optimal x:", x)

    plt.title('objective value vs. iteration')
    plt.ylabel('objective value')
    plt.xlabel('iteration')
    plt.plot(objective, 'b-o')
    plt.savefig('objective_value.png')
    plt.show()

    plt.title('Newton decrement squared divided by 2 vs. iteration')
    plt.ylabel('Newton decrement squared divided by 2')
    plt.xlabel('iteration')
    plt.plot(NT_decrement,'b-o')
    plt.yscale('log')
    plt.savefig('decrement.png')
    plt.show()

    solve_by_cvxpy(a)
    