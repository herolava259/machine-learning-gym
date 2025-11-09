import numpy as np




def pearson_correlation(x, y) -> float:

    assert len(x) == len(y)
    assert len(x.shape) == 1

    x_std = np.std(x)
    y_std = np.std(y)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std

    n = x.shape[0]

    return float(x @ y) / (n-1)