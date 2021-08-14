import numpy as np
from bound_manual_solver import b_alpha_linear, b_alpha_l2norm
from bound_Gurobi_solver import b_alpha_gurobi
from anderson import anderson
from scipy.stats import beta, uniform, randint, dirichlet, binom, bernoulli, lognorm, poisson
from helper import T_anderson



n = 6
alpha = 0.05
for i in range(10):
    z = beta(1,5).rvs(n)
    T = T_anderson(alpha, n)

    print("Sample size:",n , ", alpha:", alpha)
    print("Sample:", z)
    print("New bound with T = Anderson (Gurobi solver) and  D = [-inf,1]:", b_alpha_gurobi(z, alpha, T, lower = -np.inf))
    print("New bound with T = Anderson (manual solver) and  D = [-inf,1]:",b_alpha_linear(z, alpha, T, lower = -np.inf), "\n")

    print("New bound with T = Anderson (Gurobi solver) and  D = [0,1]:", b_alpha_gurobi(z, alpha, T, lower = 0))
    print("New bound with T = Anderson (manual solver) and  D = [0,1]:",b_alpha_linear(z, alpha, T, lower = 0), "\n")

    print("Anderson's bound when the upper bound of the support is 1: ", anderson(z, alpha), "\n")

    print("New bound with T = l2 norm (Gurobi solver) with D = [0,1]: ", b_alpha_gurobi(z,  alpha, l2norm = True))
    print("New bound with T = l2 norm (manual solver) with D = [0,1]: ", b_alpha_l2norm(z,  alpha), "\n")




