import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

from rdkit import Chem
from rdkit.Chem import AllChem

SCALE = 1
FINGERPRINT_BITS = 600

chirality = False
bond_types = False

#0.56MAE (600 bits, 2800 neurons)
#0.57MAE (500 bits, 2600 neurons)

if False and torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def mean(arr):
    thing = 0
    for i in arr:
        print(str(i) + ":" + str(thing))
        thing += 1

class LogPDataset(Dataset):
    def __init__(self, file_name):
        file = open(file_name, "r")
        lines = file.read().split("\n")
        file.close()

        x = []
        y = []

        for line in lines:
            if not line:
                continue
            
            smiles = None
            logp = None
            if "," in line:
                (smiles, logp) = line.split(",")
            else:
                (smiles, logp) = line.split(":")

            if not logp:
                continue
        
            logp = float(logp)
            if logp > 10 or logp < -10:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fingerprint = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, FINGERPRINT_BITS, chirality, bond_types))
            
                x.append(fingerprint)
                y.append([logp / SCALE])

        self.fingerprints = torch.tensor(x, dtype=torch.float32)
        self.logps = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, index):
        return (self.fingerprints[index], self.logps[index])

class PNet(nn.Module):
    def __init__(self, input_size, h1, h2, output_size):
        super().__init__()

        self.layer_one = nn.Linear(input_size, h1)
        self.layer_two = nn.Linear(h1, h2)
        self.layer_three = nn.Linear(h2, output_size)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        #x = self.dropout(x)
        x = F.relu(self.layer_one(x))
        #x = self.dropout(x)
        #x = F.relu(self.layer_two(x))
        x = self.layer_three(x)

        return x

def train(network, training_set, batch_size, optimizer, loss_function):
    network.train()

    #  creating list to hold loss per batch
    loss_per_batch = []

    #  defining dataloader
    train_loader = DataLoader(training_set, batch_size)

    #  iterating through batches
    for images, labels in train_loader:
        #---------------------------
        #  sending images to device
        #---------------------------
        images, labels = images.to(device), labels.to(device)

        #-----------------------------
        #  zeroing optimizer gradients
        #-----------------------------
        optimizer.zero_grad()

        #-----------------------
        #  classifying instances
        #-----------------------
        classifications = network(images)

        #---------------------------------------------------
        #  computing loss/how wrong our classifications are
        #---------------------------------------------------
        loss = loss_function(classifications, labels)
        loss_per_batch.append(loss.item())

        #------------------------------------------------------------
        #  computing gradients/the direction that fits our objective
        #------------------------------------------------------------
        loss.backward()

        #---------------------------------------------------
        #  optimizing weights/slightly adjusting parameters
        #---------------------------------------------------
        optimizer.step()

    return loss_per_batch

def validate(network, validation_set, batch_size, loss_function):
    #  creating a list to hold loss per batch
    loss_per_batch = []

    #  defining model state
    network.eval()

    #  defining dataloader
    val_loader = DataLoader(validation_set, batch_size)

    #  preventing gradient calculations since we will not be optimizing
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            classifications = network(images)

            loss = loss_function(classifications, labels)
            loss_per_batch.append(loss.item())
       
    return loss_per_batch

def get_error():
    with torch.no_grad():
        model.eval()
        
        length = len(validation_data)

        total = 0
        for i in range(length):
            (fingerprint, actual_logp) = validation_data[i]

            predicted_logp = model(fingerprint).item()

            total += abs(predicted_logp - actual_logp.item())

    return total / length

def get_error_real():
    with torch.no_grad():
        model.eval()
        
        length = len(real_validation_data)

        total = 0
        for i in range(length):
            (fingerprint, actual_logp) = real_validation_data[i]

            predicted_logp = model(fingerprint).item()

            total += abs(predicted_logp - actual_logp.item())

    return total / length

def predict_smiles(smiles):
    with torch.no_grad():
        model.eval()

        mol = Chem.MolFromSmiles(smiles)
        fingerprint = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, FINGERPRINT_BITS, chirality, bond_types))
        
        return model(torch.tensor(fingerprint, dtype=torch.float32)).item() * SCALE

model = PNet(FINGERPRINT_BITS, 2800, 2800, 1).to(device)

learning_rate = 0.004
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss = nn.MSELoss()

training_data = LogPDataset("training")
validation_data = LogPDataset("validation")

real_training_data = LogPDataset("realtraining")
real_validation_data = LogPDataset("realvalidation")

errors = []

epoch = 0
batch_size = 64
while True:
    try:
        training_losses = train(model, training_data, batch_size, optimizer, loss)
       
        if epoch % 1 == 0:
            validation_losses = validate(model, validation_data, batch_size, loss)
          
            training_loss = round(np.array(training_losses).mean() * SCALE, 3)
            validation_loss = round(np.array(validation_losses).mean() * SCALE, 3)
            error = round(get_error() * SCALE, 3)
            real_error = round(get_error_real() * SCALE, 3)
            aripiprazole = round(predict_smiles("C1CC(=O)NC2=C1C=CC(=C2)OCCCCN3CCN(CC3)C4=C(C(=CC=C4)Cl)Cl"), 2)

            errors.append(real_error)
            if epoch >= 3:
                old_error = errors[epoch - 3]
                improvement = old_error - real_error
                print("Improvement: {}".format(improvement))
                if improvement < 0.001:
                    print("Network reached peak performance")
                    break

            print("epoch: {}\ttraining loss: {}\tvalidation loss: {}\terror: {}\treal error: {}\taripiprazole: {}".format(epoch, training_loss, validation_loss, error, real_error, aripiprazole))

        epoch += 1
    except KeyboardInterrupt:
        break

print("Error:", get_error_real())

epoch = 0
errors = []

while True:
    try:
        training_losses = train(model, real_training_data, batch_size, optimizer, loss)
        
        if epoch % 1 == 0:
            validation_losses = validate(model, real_validation_data, batch_size, loss)
          
            training_loss = round(np.array(training_losses).mean() * SCALE, 2)
            validation_loss = round(np.array(validation_losses).mean() * SCALE, 2)
            real_error = round(get_error_real() * SCALE, 4)
            aripiprazole = round(predict_smiles("C1CC(=O)NC2=C1C=CC(=C2)OCCCCN3CCN(CC3)C4=C(C(=CC=C4)Cl)Cl"), 2)
            
            errors.append(real_error)
            if epoch >= 3:
                old_error = errors[epoch - 3]
                improvement = old_error - real_error
                print("Improvement: {}".format(improvement))
                if improvement < 0.0001:
                    print("Network reached peak performance")
                    break

            print("epoch: {}\ttraining loss: {}\tvalidation loss: {}\treal error: {}\taripiprazole: {}".format(epoch, training_loss, validation_loss, real_error, aripiprazole))
            
        epoch += 1
    except KeyboardInterrupt:
        break

print("Error:", get_error_real())

print(predict_smiles("c1ccc2c(c1)c(ncn2)Nc3cccc(c3)C(F)(F)F") - 4.09)
print(predict_smiles("c1ccc2c(c1)c(ncn2)NCc3ccc(cc3)Cl") - 	3.98)
print(predict_smiles("c1ccc(cc1)CNc2c3ccccc3ncn2") - 3.21)
print(predict_smiles("Cc1ccc2c(c1)c(c(c(=O)[nH]2)CC(=O)O)c3ccccc3") - 3.10)
print(predict_smiles("COc1cccc(c1)Nc2c3ccccc3ncn2.Cl") - 3.03)
print(predict_smiles("c1ccc(cc1)n2c3c(cn2)c(ncn3)N") - 2.10)
print(predict_smiles("c1ccc2c(c1)c(ncn2)Nc3cccc(c3)Cl.Cl") - 3.83)
print(predict_smiles("Cc1cccc(c1)Nc2c3cc(c(cc3ncn2)OC)OC") - 2.92)
print(predict_smiles("c1ccc(cc1)n2cnc3c2ccc(c3)N") - 1.95)
print(predict_smiles("c1ccc2c(c1)ncn2c3ccc(cc3)O") - 3.07)
print(predict_smiles("c1cc(c(c(c1)Cl)C(=O)Nc2ccncc2)Cl") - 2.62)
