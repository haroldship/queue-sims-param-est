import numpy as np
network_params = {
    'x0': np.array((10, 10, 10)),
    'lam': np.array((1.0, 1.0, 0.0)),
    'mu': np.array((1.0, 1.0, 1.0)),
    'C': np.array((1.0, 1.0)),  # capacity of servers 1, 2
    'c': np.array((1.0, 1.0, 1.0)),  # hold cost per item per unit time
    'G': np.array(((1.0, 0, 0), (0, 1.0, 0), (0, -1.0, 1.0)))
}