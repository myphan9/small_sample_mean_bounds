from helper import induced_mean, u_anderson
import numpy as np

def anderson(z, alpha,  upper = 1):

    u_and = u_anderson(alpha, len(z))
    return induced_mean(z,u_and, upper = upper)