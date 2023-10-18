import pywt
import matplotlib.pyplot as plt
import torch
import numpy as np

# from dataset import Dataset

plt.rcParams["figure.figsize"] = (30,6)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.color'] = 'b'
plt.rcParams['axes.grid'] = True 


def plot_signals(signals, title):
    
    plt.title("ECG signal: " + title)
    plt.plot(signals)
    plt.show()



# def denoise(data): 
#         w = pywt.Wavelet('sym4')
#         maxlev = pywt.dwt_max_level(len(data), w.dec_len)
#         threshold = 0.04 # Threshold for filtering

#         coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
#         for i in range(1, len(coeffs)):
#             coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
            
#         datarec = pywt.waverec(coeffs, 'sym4')
        
#         return datarec





def delta_encoding(data , threshold = 0.1, off_spike = False , padding = False):
    """
    Delta encoding of the data
    arguments : 

    data : list of data to encode
    threshold : threshold for the encoding
    padding : padding to add at the beginning of the data
    off_spike : for considering negative values 

    """
    delta = np.zeros((100012,360))
    

    # num_rows ,num_cols = data.shape 100012,360


    

    if not padding :
        for rows in range(0,100012) : 
            delta[rows][0] = 1

    for rows in range(100012):
        for coloums in range(0,359):
            if  data[rows][coloums+1] - data[rows][coloums]  >= threshold:
                delta[rows][coloums+1] = 1

            elif data[rows][coloums] - data[rows][coloums+1]  >= threshold and off_spike :
                delta[rows][coloums+1] = -1
                
            else :
                delta[rows][coloums+1] = 0
        
    return delta