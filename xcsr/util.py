# Brodderick Rodriguez
# Auburn University - CSSE
# July 12 2019

import os
import numpy as np
import matplotlib.pyplot as plt
import json


def load_metric(experiment_path, metric):
    results_path = experiment_path + '/results/' + metric
    contents = os.listdir(results_path)

    if len(contents) < 1:
        return

    first_repetition = np.loadtxt(results_path + '/' + contents[0], delimiter=',')
    dim = [d for d in first_repetition.shape]
    dim.insert(0, len(contents))
    data = np.zeros(dim)
    data[0] = first_repetition
    data[data == 0] = np.nan

    for i, file in enumerate(contents[1:]):
        data[i + 1] = np.loadtxt(results_path + '/' + file, delimiter=',')

    return data


def load_results(experiment_path):
    rhos = load_metric(experiment_path, 'rhos')
    predicted_rhos = load_metric(experiment_path, 'predicted_rhos')
    classifier_counts = load_metric(experiment_path, 'microclassifier_counts')
    steps = load_metric(experiment_path, 'steps')
    return rhos, predicted_rhos, classifier_counts, steps


def _get_data_single_step(experiment_path):
    rhos, pred_rhos, cc, _ = load_results(experiment_path)
    error = np.abs(rhos - pred_rhos)

    rhos_means = np.nanmean(rhos, axis=0)
    error_means = np.nanmean(error, axis=0)
    cc_means = np.nanmean(cc, axis=0) / 1000

    data = rhos_means, error_means, cc_means
    labels = 'reward', 'error', 'pop. size (/1000)'
    return data, labels


def _get_data_multi_step(experiment_path):
    rhos, pred_rhos, cc, steps = load_results(experiment_path)
    error = np.abs(rhos - pred_rhos)

    error_means = np.nanmean(np.nanmean(error, axis=0), axis=1) / 400
    cc_means = np.nanmean(np.nanmean(cc, axis=0), axis=1) / 100
    steps_means = np.nanmean(steps, axis=0)

    data = steps_means, error_means, cc_means
    labels = 'steps', 'error (/400)', 'pop. size (/100)'
    return data, labels


def _is_multi_step_environment(experiment_path):
    metadata_path = experiment_path + '/metadata.json'
    file_contents = open(metadata_path)
    json_data = json.load(file_contents)
    return json_data['is_multi_step']


def plot_results(experiment_path, interval=50, title=''):
    if _is_multi_step_environment(experiment_path):
        data, labels = _get_data_multi_step(experiment_path)
    else:
        data, labels = _get_data_single_step(experiment_path)

    _plot(data, labels, interval, title, experiment_path)


def _best_fit_line(x, y):
    x_bar = sum(x) / len(x)
    y_bar = sum(y) / len(y)
    n = len(x)

    numerator = sum([xi * yi for xi, yi in zip(x, y)]) - n * x_bar * y_bar
    denominator = sum([xi ** 2 for xi in x]) - n * x_bar ** 2

    b = numerator / denominator
    a = y_bar - b * x_bar
    y_fit = [a + b * xi for xi in x]

    lbl = 'y = {:.2f} + {:.10f}x'.format(a, b)
    plt.plot(x, y_fit, label=lbl)


def _plot(data, labels, interval, title, experiment_path):
    data_plots = [[] for _ in range(len(data) + 1)]

    for xi in range(interval, len(data[0]), interval):
        data_plots[0].append(xi / 1000)

        for j in range(len(data)):
            d = np.mean(data[j][xi - interval: xi])
            data_plots[j + 1].append(d)

    for j in range(len(data_plots) - 1):
        plt.plot(data_plots[0], data_plots[j + 1], label=labels[j])

    _best_fit_line(data_plots[0], data_plots[1])
    _best_fit_line(data_plots[0], data_plots[2])

    plt.xlabel('episodes (thousands)')
    plt.title(title)
    plt.gca().legend()
    # plt.savefig(experiment_path + '/results.png')
    plt.show()


