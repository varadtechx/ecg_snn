from utils.dataset import Dataset
from model import Net

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import itertools

from sklearn.model_selection import train_test_split




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
batch_size = 128 # batch size for dataloader 



# Load dataset into the Torch Dataloader
data_path = '/home/ubuntu/Desktop/Projects/SNN/ecg_snn/dataset/'
print("Loading dataset from: ", data_path)
dataset = Dataset()
records,annotations = dataset.load_dataset(data_path)
X = []
y = []
X, y = dataset.preprocess(records, annotations)

if X or y is not None:
    print("Signals loaded successfully..................")
    # print(len(X))


X_train, X_test, y_train, y_test = train_test_split(X,y , 
                        random_state=104,
                        train_size=0.8, shuffle=True)

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)


training_set = torch.utils.data.TensorDataset(X_train, y_train)
test_set = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = DataLoader(training_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True)

print("Dataset loaded successfully..................")


# Define network parameters
num_inputs = 128
num_hidden = 256
num_outputs = 5
num_steps = 100
beta = 0.95

LIF_Model = Net(num_hidden, num_inputs, num_outputs, num_steps, beta).to(device)
print(LIF_Model)
print("Model loaded successfully..................")

# Define loss function and optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(LIF_Model.parameters(), lr=0.0005 , betas=(0.9, 0.999))


# Train the network
num_epochs = 100
loss_hist = []
test_loss_hist = []
counter = 0
print("Starting training..................")


for epoch in range(num_epochs):
    iter_count = 0
    train_batch = iter(train_loader)

    #Minibatch training 
    for data , targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        #forward pass 
        LIF_Model.train()
        spk_rec , _ = LIF_Model(data)


        # loss and sum 
        loss_val = torch.zeros(1).to(device)
        for i in range(num_steps):
            loss_val += loss(spk_rec[:,i,:], targets)


        #Gradient descent and weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()


        # loss value storage 
        loss_hist.append(loss_val.item())




    # Test the network
    with torch.no_grad():
        LIF_Model.eval()
        test_data , test_targets = next(iter(test_loader))
        test_data = test_data.to(device)
        test_targets = test_targets.to(device)

        test_spk , text_mem = LIF_Model(test_data)

        test_loss_val = torch.zeros(1).to(device)
        for i in range(num_steps):
            test_loss_val += loss(test_spk[:,i,:], test_targets)
        test_loss_hist.append(test_loss_val.item())


        print("Epoch: ", epoch, " Iteration: ", iter_count, " Loss: ", loss_val.item(), " Test Loss: ", test_loss_val.item())



        
        iter_count += 1
        counter += 1
        
    if epoch % 10 == 0:
        torch.save({'epoch': epoch,
            'model_state_dict': LIF_Model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_hist,
            }, f'model/checkpoint_{epoch}.pth')



# Save the model
torch.save(LIF_Model.state_dict(), 'model/model.pth')

# Save the loss history
np.save('model/loss_hist.npy', loss_hist)

# Save the test loss history
np.save('model/test_loss_hist.npy', test_loss_hist)

print("Training completed..................")


# Test set accuracy
total = 0
correct = 0
with torch.no_grad():
    LIF_Model.eval()
    for data,targets in test_loader:

        data = data.to(device)
        targets = targets.to(device)

        spk_rec , _ = LIF_Model(data)

        # print(spk_rec.shape)
        # print(targets.shape)

        _, predicted = torch.max(spk_rec[:,99,:], 1)
        # print(predicted.shape)
        # print(targets.shape)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

#inferencing
# LIF_Model.load_state_dict(torch.load('model/model.pth'))
# LIF_Model.eval()
# test_data , test_targets = next(iter(test_loader))
# test_data = test_data.to(device)
# test_targets = test_targets.to(device)

# test_spk , text_mem = LIF_Model(test_data)
# print(test_spk.shape)
# print(text_mem.shape)
# print(test_targets.shape)
# print(test_spk[0,0,:])
# print(test_targets[0])
# print(text_mem[0,0,:])

# print("Inferencing completed..................")