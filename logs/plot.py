import matplotlib.pyplot as plt
import numpy as np

fontsize=44
ticksize=40
legendsize=30
plt.style.use('seaborn-white')

plt.figure(figsize=(12.8, 9.25))


def read_csv(file_name):
    return np.genfromtxt(fname=file_name, delimiter=',', skip_header=1)


def _plot_setting():
    plt.xlabel('# Epoch', fontsize=ticksize)
    plt.ylabel('Top1 - Accuracy', fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xticks(fontsize=ticksize)
    plt.legend(loc='lower right', fontsize=legendsize)
    plt.show()


# sgd = read_csv("csv/results.csv")
# plt.plot(sgd[:, 0] * 1.5, 100.0 - sgd[:, 4], 'black', label='SGD', linestyle='--', marker='s')
# hsq = read_csv("csv/val_accuracy.csv")
# plt.plot( 1.28 * (hsq[:, 0] + 1), hsq[:, 1] * 100.0, 'red', label='HSQ', linestyle='-', marker='x')
linewidth = 1
sgd = read_csv("sgd_32bit/csv/val_accuracy.csv")
plt.plot(  (sgd[:, 0] + 1), sgd[:, 1] * 100.0, 'black', label='SGD', linestyle='--', marker='x', linewidth=linewidth)
hsq = read_csv("nnq_d8_k256/csv/val_accuracy.csv")
plt.plot(  (hsq[:, 0] + 1), hsq[:, 1] * 100.0, 'red', label='HSQ', linestyle='-', marker='s', linewidth=linewidth)
_plot_setting()
