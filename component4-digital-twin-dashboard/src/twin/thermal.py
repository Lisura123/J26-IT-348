import numpy as np
from scipy.optimize import least_squares


def faiman(g, t_amb, wind, u0, u1=0.0):
    return np.asarray(t_amb, float) + np.asarray(g, float) / (u0 + u1 * np.asarray(wind, float))


def fit_faiman(g, t_amb, wind, t_module, use_wind=True):
    
    g, t_amb, wind, t_module = (np.asarray(a, float) for a in (g, t_amb, wind, t_module))

    def errors(p):
        u0 = p[0]
        u1 = p[1] if use_wind else 0.0
        return faiman(g, t_amb, wind, u0, u1) - t_module

    if use_wind:
        result = least_squares(errors, x0=[25.0, 6.84], bounds=([1, 0], [200, 50]))
        u0, u1 = result.x
    else:
        result = least_squares(errors, x0=[25.0], bounds=([1], [200]))
        u0, u1 = result.x[0], 0.0

    rmse = float(np.sqrt(np.mean(result.fun ** 2)))
    return float(u0), float(u1), rmse