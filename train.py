import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, os, sys, random, logging
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load files
parser = argparse.ArgumentParser(description='Running custom PyTorch models in Edge Impulse')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--learning-rate', type=float, required=True)
parser.add_argument('--out-directory', type=str, required=True)

args, unknown = parser.parse_known_args()

if not os.path.exists(args.out_directory):
    os.mkdir(args.out_directory)

# grab train/test set
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

classes = Y_train.shape[1]

MODEL_INPUT_SHAPE = X_train.shape[1:]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on', device)
print('')

# Small pyTorch neural network with 2 hidden layers
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        in_features = np.prod(MODEL_INPUT_SHAPE)

        # two hidden layers (20 and 10 neurons)
        self.fc1 = nn.Linear(in_features, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

# initialize the NN
model = Net()
model.to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

# convert to pyTorch float tensors
X_train = torch.FloatTensor(X_train).to(device)
Y_train = torch.FloatTensor(Y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
Y_test = torch.FloatTensor(Y_test).to(device)

# create data loaders
train_dataloader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16)
test_dataloader = DataLoader(TensorDataset(X_test, Y_test), batch_size=16)

# training loop
model.train()
for epoch in range(args.epochs):
    running_loss = 0.0
    running_loss_count = 0
    running_val_loss = 0.0
    running_val_loss_count = 0

    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_count = running_loss_count + 1

    for i, data in enumerate(test_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # validate output
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # log validation loss
        running_val_loss += loss.item()
        running_val_loss_count = running_loss_count + 1

    print(f'Epoch {epoch + 1}: loss: {running_loss / running_loss_count:.3f}, ' +
          f'val_loss: {running_val_loss / running_val_loss_count:.3f}')

# calculate accuracy
model.eval()

test_correct = 0
test_total = 0

for data, target in test_dataloader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)

    pred = pred.cpu()
    target = target.cpu()

    for i in range(len(pred)):
        if (pred[i].item() == np.argmax(target[i]).item()):
            test_correct = test_correct + 1
        test_total = test_total + 1

print('')
print('Test accuracy: %f' % (test_correct / test_total))

print('')
print('Training network OK')
print('')

# Export the model
torch.onnx.export(model.cpu(),
                  torch.randn(tuple([1] + list(X_train.shape[1:]))),
                  os.path.join(args.out_directory, 'model.onnx'),
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'])
