import snntorch as snn 
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Net(nn.Module ):
    def __init__(self, num_hidden, num_inputs, num_outputs, num_steps, beta):
        super(Net, self).__init__() 
        self.num_hidden = num_hidden
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_steps = num_steps
        self.beta = beta


        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden).to(torch.float32)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs).to(torch.float32)
        self.lif2 = snn.Leaky(beta=beta)

        

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

