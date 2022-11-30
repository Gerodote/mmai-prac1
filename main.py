import pathlib
from typing import Tuple
import re
import csv

import yaml
import numpy as np
import numpy.typing as npt
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def get_data_csv(filename:str):
    with open(filename, 'r+') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        amount_of_rows = len(list(csv_file))
        csv_file.seek(0)
        result = np.empty(shape=(amount_of_rows, 2), dtype=np.float64)
        counter_row = 0
        counter_column = 0
        for row in csv_reader:
            result[counter_row] = row
            counter_row += 1
            # except IndexError:
                # print(row)
        return result

def get_data(filename: str) -> Tuple[list, list]:
    """ get all coordinates as list of pairs from file """

    with open(filename, 'r') as file:
        content = file.read().splitlines()
    qty_of_lines_in_files = len(content)
        
    array_data = np.zeros(shape=qty_of_lines_in_files*2, dtype=np.float64)
    i:int = 0
    for line in content:
        new_line = line.replace('-', ' -')
        list_of_shit = re.split(r'\s', new_line)
        for sth in list_of_shit:
            if sth != '':
                array_data[i] = float(sth)
                i+=1
    t = np.zeros(len(array_data)//2)
    y = np.zeros(len(array_data)//2)
    for i in range(len(array_data)//2):
        t[i] = array_data[2*i]        
        y[i] = array_data[2*i + 1]

    return t, y
    
def show_graphic(t:npt.ArrayLike,y:npt.ArrayLike):  
    plt.plot(t,y)
    plt.show()

def fourier_transform(t:npt.ArrayLike,y:npt.ArrayLike):
    vector = np.dstack((t,y))[0]
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
        data = get_data_csv(filename)
        t, y = np.hsplit(data, 2)
        # t,y = get_data(filename)
        # print(x, y, sep='\n')
        show_graphic(t ,y)
        # fourier_transform(t,y)
        # result = fourier_transform(x,y)

if __name__ == '__main__':
    main()
    


