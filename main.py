import csv

import yaml
import numpy as np
import numpy.typing as npt
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


def get_data_csv(filename: str):
    with open(filename, 'r+') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        amount_of_rows = len(list(csv_file))
        csv_file.seek(0)
        data = np.empty(shape=(amount_of_rows, 2), dtype=np.float64)
        counter_row = 0
        counter_column = 0
        for row in csv_reader:
            data[counter_row] = row
            counter_row += 1
        result_x = np.array([x[0] for x in data])
        result_y = np.array([x[1] for x in data])
        return result_x, result_y


def show_graphic(t: npt.ArrayLike, y: npt.ArrayLike):
    plt.plot(t, y)
    plt.show()


def linear_interp(t, y):
    def lin_interp(x): return np.interp(x=x, xp=t, fp=y)
    result = lin_interp(t)

    plt.plot(t, y, label='data', ls='-', linewidth=5)
    plt.plot(t, lin_interp(t), label='lin_interp', linewidth=2)
    plt.legend()
    plt.show()

    return lin_interp


def fourier_transform(t: npt.ArrayLike, y: npt.ArrayLike):
    vector = np.dstack((t, y))[0]
    print(vector)
    sth = fft(vector)
    sth3 = fftfreq(20000, )
    sth2 = np.absolute(sth)
    sth3 = fft(y)
    print(sth)
    print(sth2)
    print(sth3)
    plt.plot(sth2)
    plt.show()


def main():

    with open("config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)
        filename = config['filename']
        t, y = get_data_csv(filename)
        v_interp = linear_interp(t, y)
        cfft = fourier_transform(t,y)
        # result = fourier_transform(x,y)


if __name__ == '__main__':
    main()
