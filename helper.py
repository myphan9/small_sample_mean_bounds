
import numpy as np
from scipy.stats import uniform
import numpy.ma as ma


def induced_mean(z, u,  upper = 1):
    """
    Calculate the induced mean m(z,u)
    Args:
        upper: the upper bound of the support
    Returns:
        the induced mean m(z,u)
    """
    assert len(z) == len(u)

    z1 = np.append(z,upper)
    u1 = np.append(0, u)
    u1 = np.append(u1, 1)

    z1 = np.sort(z1)
    u1 = np.sort(u1)
    n = len(u1)
    delta = u1[1:] - u1[:n-1]
    r = np.dot(delta, z1)
    return r



def upper_triangle_vertices(n, lower = 0, upper = 1):
    # Calculating the vertices of a n-dimensional polyhedron X1<=X2 <= ...<= Xn, 0<=Xi <=1 for all 1<= i <= n
    # n: the sample size
    # vertices(n) is calculated recursively from vertices(n-1) by adding 1 at the last coordinate for all vertices in vertices(n-1), and then add the vertices np.zeros(n)
    # vertices(1) = [0,1]
    # vertices(2) = [[0,0],[0,1], [1,1]]
    # vertices(3) = [[0,0,0], [0,0,1],[0,1,1] , [1,1,1] ]

    v = np.array([lower,upper])

    for i in range(2,n+1):
        v =[np.append( j,upper) for j in v]
        v = np.append([lower * np.ones(i)], v, axis = 0)
    return np.array(v)

def vertices_with_intersection(n, lower, upper, T, z, degree):
    """ Find all vertices of the n-dimensional polyhedron defined by x1<=x2 <= ...<= xn, lower<=Xi <=upper for all 1<= i <= n
              and <T,x**degree> <= <T,z**degree> where x**degree denote element-wise operation.
        Find all intersection of the plane <T,x**degree> = <T,z**degree> with the above polyhedron
    Args:
        n: the dimension of the polyhedron
        lower, upper: lower<=xi <=upper
        T, Tz, degree: <T,x**degree> <= <T,z**degree>
    Output:
        The vertices of the polyhedron and all intersection of the plane <T,x**degree> = <T,z**degree> with it.
    """
    T = np.array(T)
    z_squared = np.power(z,degree)
    Tz= np.dot(z_squared, T)
    #List of vertices of the n-dimensional polyhedron X1<=X2 <= ...<= Xn, 0<=Xi <=1 for all 1<= i <= n
    DUMMY_INF = -10000
    if lower == -np.inf:
        vertices = upper_triangle_vertices(n, DUMMY_INF, upper)
    else:
        vertices = upper_triangle_vertices(n, lower, upper)

    vertices_squared = np.power(vertices, degree)


    Tv = vertices_squared @ T - Tz
    # Add all vertices v of the polyhedron such that T(v) <= Tz to the candidate
    candidates = vertices[Tv <= 0]
    # Loop through all edges of the n-dimensional polyhedron X1<=X2 <= ...<= Xn, 0<=Xi <=1 for all 1<= i <= n
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            vi = vertices[i]
            vj = vertices[j]
            add_intersection = False
            if lower != - np.inf:
                if Tv[i] * Tv[j]   <= 0: # The interval (vi, vj) intersects with the plane T(x) = T(z)
                    add_intersection = True
            else:
                # if the edge is in the plane vi[0] == DUMMY_INF, there is no need to intersect
                if not (vi[0] == DUMMY_INF and vj[0] == DUMMY_INF):
                    add_intersection = True
            if add_intersection:
                # If vi is [0, 0, 0, 1, 1] and vj is [0, 1, 1, 1, 1] then the line (vi,vj) has the form [0, t, t, 1,1] where 0<= t <= 1 is a real number
                # We set line_direction = [0, MASKED, MASKED, 1,1]
                line_direction = ma.masked_array(vi, mask = vi != vj)
                # Calculate the value of t so that the point [0, t, t, 1,1] is in the plane T(x) = T(z)
                # In the example [0, t, t, 1,1]:
                # T(z) = T[0]*0 + (T[1]+T[2])* t + T[3]*1 + T[4]*1
                # Therefore:
                # t = (T(z) - (T[0]*0 + T[3]*1 + T[4*1]))/(T[1]+T[2])
                # Find the sum of the coefficient T[1] + T[2] at the masked values
                masked_coeff = np.sum(T[ma.getmask(line_direction)])
                if masked_coeff !=0:
                    rr = (Tz - np.dot(T, np.power(line_direction.filled(fill_value = 0), degree)))/masked_coeff
                    if (degree == 2 and rr >=0) or degree == 1:
                        t =  rr** (1/degree)
                        # Fill the masked value by t
                        intersection = line_direction.filled(fill_value = t)
                        # Add the intersection to the list of candidates
                        candidates = np.append(candidates, [intersection], axis = 0)
                elif Tz - np.dot(T, np.power(line_direction, degree)) == 0:
                    # the edge is in the plane (T(x) = T(z))
                    intersection_min = line_direction.filled(fill_value = lower)
                    intersection_max = line_direction.filled(fill_value = upper)
                    candidates = np.append(candidates, [intersection_min, intersection_max], axis = 0)

    return candidates

def u_anderson(alpha, sample_size):
    """ Computing u_anderson
    Args:
        alpha: confidence parameter
        sample_size: the sample size
    Output:
        u_anderson
    """
    Fn = [(i+1)/sample_size for i in range(sample_size)]
    Us = uniform().rvs((1000000, sample_size))
    Us = np.sort(Us, axis = 1)
    Us =  Fn - Us
    r = np.max(Us, axis = 1)
    assert np.all(r>=0)
    r = np.sort(r)
    q = r[int(np.ceil((1-alpha)*len(r)))]

    u_and= [max(0, (i+1)/sample_size - q) for i in range(sample_size)]
    return u_and

def T_anderson(alpha, n):
    """ Computing the linear coefficient of the function T = Anderson
    Args:
        alpha: confidence parameter
        sample_size: the sample size
    Output:
        the linear coefficient of the function T = Anderson
    """

    u_and = u_anderson(alpha, n)
    u_and = np.append(0, u_and)
    delta_u = u_and[1:] - u_and[:n]
    return delta_u