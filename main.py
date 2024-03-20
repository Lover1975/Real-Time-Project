import numpy as np


def UUniFast(n, u_bar):
    sum_u = u_bar
    vect_u = np.zeros(n)
    for i in range(n - 1):
        next_sum_u = sum_u * (1 - (np.random.rand() ** (1.0 / (n - i))))
        vect_u[i] = sum_u - next_sum_u
        sum_u = next_sum_u
    vect_u[n - 1] = sum_u
    return vect_u
