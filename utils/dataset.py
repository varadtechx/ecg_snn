import numpy as np 
import pandas as pd 
import os

import matplotlib.pyplot as plt
import csv
import itertools
import collections

import pywt
from scipy import stats

# from utils import denoise

plt.rcParams["figure.figsize"] = (30,6)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.color'] = 'b'
plt.rcParams['axes.grid'] = True 




class Dataset:
    def __init__(self):
        self.dataset = None
        self.data_path = None


    def denoise(data): 
        w = pywt.Wavelet('sym4')
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        threshold = 0.04 # Threshold for filtering

        coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
            
        datarec = pywt.waverec(coeffs, 'sym4')
        
        return datarec

    def load_dataset(self, data_path):

        print("Loading dataset from: ", data_path)
        
        records = list()
        annotations = list()

        for file in os.listdir(data_path):
            if file.endswith(".csv"):
                records.append(data_path+file)
            elif file.endswith(".txt"):
                annotations.append(data_path+file)

        records.sort()
        annotations.sort()

        # print(records[0], annotations[0])

        return records, annotations

        


    def preprocess(self,records,annotations):
        maximum_counting = 10000
        window_size = 180
        classes = ['N', 'L', 'R', 'A', 'V']
        n_classes = 5
        count_classes = [0]*5
        X = list()
        y = list()
        for r in range(0,len(records)):
            signals = []

            with open(records[r], 'rt') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|') # read CSV file\
                row_index = -1
                for row in spamreader:
                    if(row_index >= 0):
                        signals.insert(row_index, int(row[1]))
                    row_index += 1
                    
            # # Plot an example to the signals
            # if r == 6:
            #     # Plot each patient's signal
            #     plt.title(records[6] + " Wave")
            #     plt.plot(signals[0:700])
            #     plt.show()
            def denoise(data): 
                w = pywt.Wavelet('sym4')
                maxlev = pywt.dwt_max_level(len(data), w.dec_len)
                threshold = 0.04 # Threshold for filtering

                coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
                for i in range(1, len(coeffs)):
                    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
                    
                datarec = pywt.waverec(coeffs, 'sym4')
                
                return datarec
            signals = denoise(signals)
            # # Plot an example to the signals
            # if r == 6:
            #     # Plot each patient's signal
            #     plt.title(records[6] + " wave after denoised")
            #     plt.plot(signals[0:700])
            #     plt.show()
                
            # signals = stats.zscore(signals)
            # # Plot an example to the signals
            # if r == 6:
            #     # Plot each patient's signal
            #     plt.title(records[6] + " wave after z-score normalization ")
            #     plt.plot(signals[0:700])
            #     plt.show()
            
            # Read anotations: R position and Arrhythmia class
            example_beat_printed = False
            with open(annotations[r], 'r') as fileID:
                data = fileID.readlines() 
                beat = list()

                for d in range(1, len(data)): # 0 index is Chart Head
                        splitted = data[d].split(' ') 
                        splitted = filter(None, splitted)
                        next(splitted) # Time... Clipping
                        pos = int(next(splitted)) # Sample ID
                        arrhythmia_type = next(splitted) # Type
                        if(arrhythmia_type in classes):
                            arrhythmia_index = classes.index(arrhythmia_type)
                    # if count_classes[arrhythmia_index] > maximum_counting: # avoid overfitting
                            #    pass
                        #else:
                            count_classes[arrhythmia_index] += 1
                            if(window_size <= pos and pos < (len(signals) - window_size)):
                                beat = signals[pos-window_size:pos+window_size]     ## REPLACE WITH R-PEAK DETECTION
                                # Plot an example to a beat    
                                # if r == 6 and not example_beat_printed: 
                                #     plt.title("A Beat from " + records[6] + " Wave")
                                #     plt.plot(beat)
                                #     plt.show()
                                #     example_beat_printed = True

                                X.append(beat)
                                y.append(arrhythmia_index)

        return X, y