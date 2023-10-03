import pywt
import matplotlib.pyplot as plt

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