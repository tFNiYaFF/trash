import Model
import Analyze
import matplotlib.pyplot as plt
import numpy as np


def task4():
    n = 1000
    k = 0.1
    b = 1000
    max_value = 100
    min_value = -100
    x_number = np.arange(0, n, 1)

    random_values = Model.init_random_array_using_rnd_func(n, max_value, min_value, Model.builtin_random)

    linear_values0 = [Model.linear_function(k, b, x) for x in range(n)]
    linear_values1 = [Model.linear_function(-k, b, x) for x in range(n)]

    additive_model0 = Model.add_trends(random_values, linear_values0, n)
    additive_model1 = Model.add_trends(random_values, linear_values1, n)

    multiplicative_model0 = Model.multiply_trends(random_values, linear_values0, n)
    multiplicative_model1 = Model.multiply_trends(random_values, linear_values1, n)

    plt.figure()

    plt.subplot(211)
    plt.plot(x_number, additive_model0, label="k > 0")
    plt.legend(title="additive")

    plt.subplot(212)
    plt.plot(x_number, additive_model1, label="k < 0")
    plt.legend(title="additive")
    plt.show()

    plt.figure()
    plt.subplot(211)
    plt.plot(x_number, multiplicative_model0, label="k > 0")
    plt.legend(title="multiplicative")
    plt.subplot(212)
    plt.plot(x_number, multiplicative_model1, label=" k < 0")
    plt.legend(title="multiplicative")
    plt.show()

    impl_num = 1
    random_values = []
    for i in range(impl_num):
        random_values.append(Model.init_random_array_using_rnd_func(n, max_value, min_value, Model.builtin_random))
    random_values = np.sum(random_values, 0) / impl_num
    avg_value = Model.interval_average(0, n, random_values)
    print("dispersion for n = ", impl_num)
    print(np.sqrt(Model.interval_dispersion(0, n, random_values, avg_value)))

    impl_num = 10
    random_values = []
    for i in range(impl_num):
        random_values.append(Model.init_random_array_using_rnd_func(n, max_value, min_value, Model.builtin_random))
    random_values = np.sum(random_values, 0) / impl_num
    avg_value = Model.interval_average(0, n, random_values)
    print("dispersion for n = ", impl_num)
    print(np.sqrt(Model.interval_dispersion(0, n, random_values, avg_value)))


def task3():
    n = 1000
    m = 10
    k = 1
    b = 0
    max_value = 100
    min_value = -100
    x_number = np.arange(0, m, 1)

    builtin_values = Model.init_random_array_using_rnd_func(n, max_value, min_value, Model.builtin_random)
    builtin_averages = Model.intervals_average(n, m, builtin_values)
    builtin_dispersions = Model.intervals_dispersion(n, m, builtin_values, builtin_averages)

    custom_values = Model.init_random_array_using_rnd_func(n, max_value, min_value, Model.custom_random)
    custom_averages = Model.intervals_average(n, m, custom_values)
    custom_dispersions = Model.intervals_dispersion(n, m, custom_values, custom_averages)

    trend_values = [Model.linear_function(k, b, x) for x in range(n)]
    trend_averages = Model.intervals_average(n, m, trend_values)
    trend_dispersions = Model.intervals_dispersion(n, m, trend_values, trend_averages)

    builtin_averages_diffs = Model.get_average_diff(m, max_value * 2, builtin_averages)
    custom_averages_diffs = Model.get_average_diff(m, max_value * 2, custom_averages)
    trend_averages_diffs = Model.get_average_diff(m, np.max(trend_values) - np.min(trend_values), trend_averages)

    builtin_dispersions_diffs = Model.get_dispersion_diff(m, builtin_dispersions)
    custom_dispersions_diffs = Model.get_dispersion_diff(m, custom_dispersions)
    trend_dispersions_diffs = Model.get_dispersion_diff(m,  trend_dispersions)

    print("BUILTIN AVERAGES DIFFS")
    print(builtin_averages_diffs)
    print("CUSTOM AVERAGES DIFFS")
    print(custom_averages_diffs)
    print("TREND AVERAGES DIFFS")
    print(trend_averages_diffs)

    print("BUILTIN DISPERSION DIFFS")
    print(builtin_dispersions_diffs)
    print("CUSTOM DISPERSION DIFFS")
    print(custom_dispersions_diffs)
    print("TREND DISPERSION DIFFS")
    print(trend_dispersions_diffs)

    plt.plot(x_number, custom_averages, label="custom random")
    plt.plot(x_number, builtin_averages, label="builtin random")
    plt.legend(title="averages")
    plt.show()

    plt.plot(x_number, builtin_dispersions, label="custom random")
    plt.plot(x_number, custom_dispersions, label="builtin random")
    plt.legend(title="dispersions")
    plt.show()


def task2():
    n = 1000
    m = 10
    s = 100
    scale = 10
    k = 1
    b = 1

    x_number = np.arange(0, n, 1)
    linear_values = [Model.linear_function(k, b, x) for x in range(n)]

    spike_array = Model.spike(n, m, scale, s)
    shift_array = Model.shift_interval(n, 200, 600, x_number, linear_values)

    plt.figure()

    plt.subplot(211)
    plt.plot(x_number, linear_values, label="original linear")
    plt.legend()

    plt.subplot(212)
    plt.plot(x_number, shift_array, label="shifted linear")
    plt.legend()

    plt.show()

    plt.plot(x_number, spike_array, label="spike")
    plt.legend()
    plt.show()


def task1():
    n = 1000
    k = 1
    b = 1
    alpha = 0.01
    beta = 1
    delta = 1
    start_point = 0

    t0 = np.arange(start_point, delta * n, delta)
    t1 = np.arange(start_point, delta * n, delta)
    t2 = np.arange(start_point, delta * n, delta)
    t3 = np.arange(start_point, delta * n, delta)

    plt.figure()

    plt.subplot(221)
    plt.plot(t0, Model.linear_function(k, b, t0), label="k > 0")
    plt.legend(title="linear")

    plt.subplot(222)
    plt.plot(t1, Model.linear_function(-k, b, t1), label="k < 0")
    plt.legend(title="linear")

    plt.subplot(223)
    plt.plot(t2, Model.exp_function(beta, alpha, t2), label="alpha > 0")
    plt.legend(title="exp")

    plt.subplot(224)
    plt.plot(t3, Model.exp_function(beta, -alpha, t3), label="alpha < 0")
    plt.legend(title="exp")

    plt.show()

    plt.plot(t0, Model.get_common_function(k, b, alpha, beta, n), label="3 functions")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()

