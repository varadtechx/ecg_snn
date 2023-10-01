import pywt
import matplotlib.pyplot as plt

# from dataset import Dataset




def plot_signals(signals, title):
    
    plt.title("ECG signal: " + title)
    plt.plot(signals)
    plt.show()