
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from helper import induced_mean



def b_alpha_gurobi(z, alpha,  T = None, l2norm = False, lower = 0, upper = 1, num_samples = 10000):
    """
    Args:
      z: The samlpe
      alpha: confidence parameter
      T: the coefficient of the linear function mapping the sample z to a real number
      l2norm: if l2norm is true, the function mapping the sample z to a real number is the l2 norm of z.
      lower: lower bound of the support
      upper: upper bound of the support
      num_samples: the number of Monte Carlo samples

    Examples:

      When the functino is the sample mean, T is a vector of 1.

    Returns:
      the value of the bound when the function T is a linear function.
    """
    z = np.sort(z)
    n = len(z)
    b_list = []

    u_list = np.random.rand(num_samples, n)

    sorted_u_list = np.sort(u_list, axis = 1)
    sorted_u_list0 = np.append(np.zeros((num_samples, 1)), sorted_u_list, axis = 1)
    neg_u_delta_list = sorted_u_list0[:, :n] - sorted_u_list0[:, 1:]


    model = gp.Model()
    model.Params.LogToConsole = 0
    # Constant term of the function f(x). This is a free continuous variable that can take positive and negative values.
    x = model.addMVar(n, lb=lower , ub=upper, vtype=GRB.CONTINUOUS, name="x")
    order = model.addConstrs( (x[i] - x[i+1] <= 0 for i in range(n-1)), name='order')
    if not l2norm:
        assert T is not None
        T = np.array(T)
        region = model.addConstr( (x @ T - z @ T <= 0) , name='region')
    else:
        region = model.addQConstr( (x @ x - z @ z <= 0) , name='region')


    for j in range(len(u_list)):
        u = u_list[j]
        neg_u_delta = neg_u_delta_list[j]
        model.setObjective(x @ neg_u_delta)
        model.optimize()


        if model.status == GRB.OPTIMAL:

            x_opt = x.x

            m_opt = induced_mean(x_opt,u, upper = upper)
            b_list.append(m_opt)
        else:
            print("GUROBI does not return optimal solution. Status: " + str(model.status))
            break


    sorted = np.sort(b_list)
    r = sorted[int(np.ceil((1-alpha)*len(sorted)))]

    return r
