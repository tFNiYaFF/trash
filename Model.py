import numpy as np
import random
import calendar
import time


def linear_function(k, b, x):
    return k * x + b


def exp_function(beta, alpha, x):
    return beta * np.power(np.e, alpha * x)


def get_common_function(k, b, alpha, beta, size):
    n = int(np.floor(size / 3))
    linear_values0 = [linear_function(k, b, x) for x in range(n)]
    linear_values1 = [linear_function(-k, b + 333, x) for x in range(n)]
    if n * 3 != size:
        n += 1
    exp_values0 = [exp_function(beta, alpha, x) for x in range(n)]
    return linear_values0 + linear_values1 + exp_values0


def builtin_random(min_value, max_value):
    return random.uniform(min_value, max_value)


def custom_random(min_value, max_value):
    if "current" not in custom_random.__dict__:
        custom_random.current = calendar.timegm(time.gmtime())
    a = 1103515245
    m = 2 ** 32
    c = 12345
    custom_random.current = (a * custom_random.current + c) % m
    return min_value + (max_value - min_value) * (custom_random.current / m)


def init_random_array_using_rnd_func(n, max_value, min_value, rnd_func):
    arr = []
    for i in range(n):
        arr.append(rnd_func(min_value, max_value))
    return arr


def shift_interval(n, a, b, x_values, y_values):
    new_values = []
    for i in range(len(x_values)):
        if a <= x_values[i] <= b:
            new_values.append(y_values[i] + n)
        else:
            new_values.append(y_values[i])
    return new_values


def spike(n, m, scale, s):
    arr = [0] * n
    for i in range(m):
        rnd_index = int(builtin_random(0, n-1))
        while arr[rnd_index] != 0:
            rnd_index = int(builtin_random(0, n - 1))
        arr[rnd_index] = scale * builtin_random(-s, s)
    return arr


def interval_average(start, finish, values):
    acc = 0
    for i in range(start, finish, 1):
        acc += values[i]
    return acc / (finish - start)


def interval_dispersion(start, finish, values, avg):
    acc = 0
    for i in range(start, finish, 1):
        acc += np.power(values[i] - avg, 2)
    return acc / (finish - start)


def intervals_average(n, m, values):
    average_values = []
    diff = int(n / m)
    for i in range(m):
        start = i * diff
        finish = (i + 1) * diff
        average_values.append(interval_average(start, finish, values))
    return average_values


def intervals_dispersion(n, m, values, average_values):
    dispersion_values = []
    diff = int(n / m)
    for i in range(m):
        start = i * diff
        finish = (i + 1) * diff
        dispersion_values.append(interval_dispersion(start, finish, values, average_values[i]))
    return dispersion_values


def get_average_diff(n, interval, values):
    diffs = []
    for i in range(n):
        if i == 0:
            diffs.append(np.abs(values[i]) / interval)
        else:
            diffs.append(np.abs(values[i] - values[i-1]) / interval)
    return diffs


def get_standard_deviation(n, dispersion_values):
    standard_deviation = []
    for i in range(n):
        standard_deviation.append(np.sqrt(dispersion_values[i]))
    return standard_deviation


def get_dispersion_diff(n, dispersion_values):
    dispersion_diffs = []
    for i in range(n):
        if i == 0:
            dispersion_diffs.append(0)
        else:
            dispersion_diffs.append(np.abs(dispersion_values[i] - dispersion_values[i-1]) / dispersion_values[i-1])
    return dispersion_diffs


def add_trends(trend_one, trend_two, n):
    result_trend = []
    for i in range(n):
        result_trend.append(trend_one[i] + trend_two[i])
    return result_trend


def multiply_trends(trend_one, trend_two, n):
    result_trend = []
    for i in range(n):
        result_trend.append(trend_one[i] * trend_two[i])
    return result_trend

