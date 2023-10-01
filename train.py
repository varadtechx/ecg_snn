from utils.dataset import Dataset



data_path = '/home/ubuntu/Desktop/Projects/SNN/dataset/'
dataset = Dataset()
records, annotations = dataset.load_dataset(data_path)

X = []
y = []
X, y = dataset.preprocess(records, annotations)

if X or y is not None:
    print("Dataset loaded successfully..................")

# print(X[0])
# print(y[1:10])