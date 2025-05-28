import numpy as np
import cvxpy as cp

np.random.seed(901052)

n = 50
m = 100
lambda_ = 1
beta = cp.Variable(n)

X = np.random.randn(n, m)

y = np.random.randint(0, 2, size=m)
print("y shape:", y.shape)

# objective = cp.Minimize(
#     cp.sum(cp.logistic(cp.multiply(-y, (X.T @ beta))))/m + lambda_ * cp.square(cp.norm(beta, 2))
# )
loss = cp.sum(cp.logistic(cp.multiply(-y, (X.T @ beta))))/m
objective = cp.Minimize(loss + lambda_ * cp.square(cp.norm(beta, 2)))

problem = cp.Problem(objective)
problem.solve()

print("Optimal value:", problem.value)
print("Optimal beta:", beta.value)
np.savetxt("X.csv", X, delimiter=",", fmt="%.5f")
np.savetxt("y.csv", y, delimiter=",", fmt="%.5f")
np.savetxt("beta.csv", beta.value, delimiter=",", fmt="%.5f")