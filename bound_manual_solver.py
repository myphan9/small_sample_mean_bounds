
import numpy as np
from numpy.linalg import norm
from helper import induced_mean, vertices_with_intersection


def b_alpha_linear_inner(z, alpha,  T, lower = 0, upper = 1,  num_samples = 10000, degree=1 ):
    """
    Args:
      z: The samlpe
      alpha: confidence parameter
      T: the coefficient of the linear function mapping the sample z to a real number
      lower: lower bound of the support
      upper: upper bound of the support
      num_samples: the number of Monte Carlo samples
      degree: the function mapping the sample z to a real number is defined as the dot product <T, z**degree> where z**degree is element-wise operation.
    Examples:
      When the function is the l2 norm, T is a vector of 1, and degree = 2.
      When the functino is the sample mean, T is a vector of 1, and degree = 1.

    Returns:
      b_list: For each Monte Carlo sample u, return the maximum b(x,u) over all points in the n-dimensional polyhedron x1<=x2 <= ...<= xn, lower<=Xi <=upper for all 1<= i <= n
              and <T,x> <= <T,z>
      u_delta_list: the list of [u_{i}- u_{i-1} ,1<=i <= n+1] for all Monte Carlo samples u (this is for when T is the l2 norm)

    """

    n = len(z)
    z = np.sort(z)
    b_list = []
    u_list = np.random.rand(num_samples, n)

    #Add U_0 = 0
    u_list0 = np.append(np.zeros((num_samples, 1)), u_list, axis = 1)
    #Add U_{n+1} = 1
    u_list01 = np.append(u_list0, np.ones((num_samples, 1)), axis = 1)
    sorted_u_list = np.sort(u_list01, axis = 1)
    #Calculate the list of [u_{i}- u_{i-1} ,1<=i <= n+1] for all sample U
    u_delta_list = sorted_u_list[:, 1:] - sorted_u_list[:, :n+1]


    candidates = vertices_with_intersection(n, lower, upper,  T, z, degree)

    # Add upper to the vertices in the candidate set
    candidates1 = np.append(candidates, upper * np.ones((candidates.shape[0], 1)), axis = 1)
    sorted_candidates1 = np.sort(candidates1, axis = 1)
    # Calculate m(x,U) = <x, delta_u>
    b_mat = sorted_candidates1 @ u_delta_list.T
    # For each u, take max_x m(x,u)
    b_list = b_mat.max(axis = 0)

    return b_list, u_delta_list

def b_alpha_linear(z, alpha,  T , lower = 0, upper = 1,  num_samples = 10000):
    """ This function calculuates the value of the bound when the function T is a linear function.
    Args:
      z: The samlpe
      alpha: confidence parameter
      T: the coefficient of the linear function mapping the sample z to a real number
      lower: lower bound of the support
      upper: upper bound of the support
      num_samples: the number of Monte Carlo samples
    Examples:
      When the functino is the sample mean, T is a vector of 1.

    Returns:
      the value of the bound when the function T is a linear function.
    """
    degree = 1
    b_list, _ = b_alpha_linear_inner(z, alpha,  T , lower , upper,  num_samples, degree)
    sorted = np.sort(b_list)
    r = sorted[int(np.ceil((1-alpha)*len(sorted)))]
    return r

def b_alpha_l2norm(z, alpha,  upper = 1, num_samples = 10000):
    """ This function calculuates the value of the bound when the function T is the l2 norm and the lower bound of the support is 0.
    Args:
      z: The samlpe
      alpha: confidence parameter
      T: the coefficient of the linear function mapping the sample z to a real number
      upper: upper bound of the support
      num_samples: the number of Monte Carlo samples
    Examples:
      When the function is the l2 norm, T is a vector of 1, and degree = 2.
      When the function is the sample mean, T is a vector of 1, and degree = 1.

    Returns:
      the value of the bound when the function T is the l2 norm and the lower bound of the support is 0.
    """
    lower = 0
    n = len(z)
    T = np.ones(n)

    b_list, u_delta_list = b_alpha_linear_inner(z, alpha, T, lower = lower, upper = upper, num_samples = num_samples, degree = 2)

    ascending = np.all(u_delta_list[:,:n-1] - u_delta_list[:, 1:n] <=0, axis =1)
    intersection = u_delta_list[:, :n] / norm(u_delta_list[:, :n], axis = 1)[:, None] * norm(z)

    intersection_bounded = np.all(intersection <= upper, axis = 1)
    in_region = np.logical_and(ascending,intersection_bounded)

    intersection1 = np.append(intersection, upper* np.ones((intersection.shape[0],1)), axis = 1)

    b_u_intersection = np.einsum('ij,ij->i',intersection1 , u_delta_list)

    b_list[in_region] = np.maximum(b_list[in_region], b_u_intersection[in_region])
    sorted = np.sort(b_list)
    r = sorted[int(np.ceil((1-alpha)*len(sorted)))]

    return r
