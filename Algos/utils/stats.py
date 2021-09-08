import numpy as np
import scipy


def rmse(y, y_pred):
    return np.sqrt(((y_pred - y) ** 2).mean())


def r2(y, y_pred):
    sse_mean = np.sum((y - np.mean(y))**2)  # variation (sum of squared error) around the mean of y
    sse_fit = np.sum((y_pred - y)**2)  # variation (sum of squared error) around the predicted y
    return (sse_mean - sse_fit)/sse_mean


def f_test(y, y_pred, df1, df2):
    sse_mean = np.sum((y - np.mean(y))**2)  # variation (sum of squared error) around the mean of y
    sse_fit = np.sum((y_pred - y)**2)  # variation (sum of squared error) around the predicted y
    var_explain_by_extra_param = (sse_mean - sse_fit)/df1
    var_not_explain_by_extra_param = sse_fit/df2
    f_value = var_explain_by_extra_param/var_not_explain_by_extra_param
    p_value = 1 - scipy.stats.f.cdf(f_value*(df1/df2), df1, df2)
    return f_value, p_value


