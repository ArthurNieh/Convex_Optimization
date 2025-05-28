import matplotlib.pyplot as plt
import numpy as np
# This code plots the feasible region for the linear inequalities:
# x1 + x2 <= 2
# x1 >= 1
# x2 >= 0
# The feasible region is the area where all inequalities are satisfied.

x1 = np.linspace(1, 2, 100)
x2 = 2 - x1
plt.plot(x1, x2, label='x1+x2=2')
plt.xlim(0, 3)
plt.ylim(0, 3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Feasible Region')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.grid()
plt.legend()
plt.savefig('2d_feasible_region.png')
plt.show()
