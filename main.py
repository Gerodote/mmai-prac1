import csv

import yaml
import numpy as np
import numpy.typing as npt
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
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


# def linear_interp(t, y):
#     def lin_interp(x): return np.interp(x=x, xp=t, fp=y)
#     result = lin_interp(t)

#     plt.plot(t, y, label='data', ls='-', linewidth=5)
#     plt.plot(t, lin_interp(t), label='lin_interp', linewidth=2)
#     plt.legend()
#     plt.show()

#     return lin_interp


def fourier_transform(t: npt.ArrayLike, y: npt.ArrayLike):
    difference = t[1] - t[0]
    freq = fftfreq(len(y), d=difference)
    result_of_fft = fft(y)
    plt.plot(freq[:len(freq)//2], np.abs(result_of_fft)[:len(freq)//2])
    plt.show()

    plt.plot(freq[:len(freq)//8], np.abs(result_of_fft)[:len(freq)//8])
    plt.show()
    
    plt.plot(freq[:len(freq)//8], np.arctan2(np.imag(result_of_fft),np.real(result_of_fft))[:len(freq)//8])
    plt.show()
    
    return freq, difference, result_of_fft

def find_first_k_argmax(array:np.ndarray, k):
    ind = np.argpartition(array, -k)[-k:]
    ind_sorted_top_k = ind[np.argsort(array[ind])][::-1]
    print(ind_sorted_top_k)
    print(array[ind_sorted_top_k])
    return ind_sorted_top_k
    

def func_to_fit_to(x, A, r, freq, phase):
    return A*np.exp(-r*x)*np.sin(2*np.pi*freq*x + phase)


    
def main():

    with open("config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)
        filename = config['filename']
        t, y = get_data_csv(filename)
        y_norm = y/np.max(y)
        show_graphic(t, y_norm)
        freq, difference, result_of_fft = fourier_transform(t,y_norm)
        how_many_peaks = 4
        ind_max = find_first_k_argmax(np.abs(result_of_fft)[:len(result_of_fft)//2],how_many_peaks)        
        phases = np.arctan2(np.imag(result_of_fft),np.real(result_of_fft))
        needed_freq = freq[:len(freq)//2][ind_max]
        needed_phases = phases[:len(freq)//2][ind_max]
        print(needed_freq)
        print(needed_phases)
        foos = [lambda x, A, r: func_to_fit_to(x, A, r, needed_freq[i], needed_phases[i]) for i in range(len(needed_phases))]
        model = lambda x,  A0, r0, A1, r1, A2, r2, A3, r3: foos[0](x, A0, r0) + foos[1](x, A1, r1) + foos[2](x, A2, r2) + foos[3](x, A3, r3)
        popt , pcov = curve_fit(f=model, xdata=t, ydata=y_norm, p0=(1,0.1,1,0.1,1,0.1,1,0.1), maxfev=10000)
        print(popt, pcov)
        
        
        

if __name__ == '__main__':
    main()
