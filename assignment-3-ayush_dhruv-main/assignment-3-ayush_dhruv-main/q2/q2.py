from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split


class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,num_classes)
    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

class DatasetLoader(Dataset):
    def __init__(self,X, ground_truth):
        self.X= X
        self.y = ground_truth
    def __len__(self):
        return len(self.y)
    def __getitem__(self,index):
        # ground_truth is a torch tensor that has the values
        return (self.X[index],self.y[index])
    

        


# data = torch.rand(200,50)
rng = np.random.RandomState(0)
X = rng.randn(200, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
Y = np.multiply(Y, 1)
# print(Y)


# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# X_train = torch.tensor(X_train)
# y_train = torch.tensor(y_train)
# X_test = torch.tensor(X_test)
# y_test = torch.tensor(y_test)
# print(f"X is {X.shape}")
# model = NN(input_size=2,num_classes=2)
# y = model(X.float())
# print(y.shape)

# optimizer = nn.optim.Adam()
batch_size = 20
num_epochs = 100
input_size = 2
num_classes = 1
learning_rate = 1e-3
dataset = DatasetLoader(X = X, ground_truth=Y)
train_set,test_set = torch.utils.data.random_split(dataset,[180,20])
train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset = test_set,batch_size=batch_size,shuffle=True)
model = NN(input_size=input_size,num_classes=num_classes)
criterion = nn.BCELoss()
optimizer =optim.Adam(model.parameters(),lr = learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for epoch in range(num_epochs):
    for idx, (data,outputs) in enumerate(train_loader):
        data  =  data.to(device)
        outputs = outputs.to(device)
        # print(data.shape)
        predictions = model(data.float())
        predictions = torch.squeeze(predictions)
        # print(f"predictions ka shape is {predictions.shape}")
        # predictions = torch.argmax(predictions,dim=1)
        # print(predictions)
        # predictions = predictions.to(torch.float32)
        outputs = outputs.to(torch.float32)
        # print(f"final preds is {predictions} outputs are {outputs}")
        loss = criterion(predictions,outputs)
        # print("abc")
        optimizer.zero_grad()
        # print("cde")
        # print(loss)
        loss.backward()
        # print("fgh")
        optimizer.step()
    print(f"finished epoch number {epoch+1} of training")
    print(f"loss value after epoch {epoch+1} ---> {loss}")
def accuracy_check(loader,model,set):
    # if(loader.dataset.train):
        # print("Accuracy being checked on training data")
    
    print(f"Accuracy being checked on {set} data")
    no_correct = 0
    no_samples = 0
    model.eval()
    with torch.no_grad():
        for data,output in loader:
            preds = model(data.float())
            preds = torch.squeeze(preds)
            # print(preds)
            final_preds = []
            for pred in preds:
                if(pred<0.5):
                    final_preds.append(0)
                else:
                    final_preds.append(1)
            final_preds = torch.tensor(final_preds)
            no_correct += (final_preds==output).sum()
            no_samples+=len(final_preds)
            accuracy = float(no_correct)/float(no_samples)*100

            print(f"Got {no_correct}/{no_samples} with accuracy {accuracy:.2f}%")
    model.train()


print("came here")
accuracy_check(train_loader,model,"train")         
accuracy_check(test_loader,model,"test")

