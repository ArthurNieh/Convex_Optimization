import cvxpy as cp

x1 = cp.Variable()
x2 = cp.Variable()

objective = cp.Minimize(cp.square(x1 + x2 - 1))
constraints = [x1 + x2 == 2, x1 >= 1, x2 >= 0]

problem = cp.Problem(objective, constraints)
problem.solve()
print("Optimal value:", problem.value)
print("Optimal x1:", x1.value)
print("Optimal x2:", x2.value)