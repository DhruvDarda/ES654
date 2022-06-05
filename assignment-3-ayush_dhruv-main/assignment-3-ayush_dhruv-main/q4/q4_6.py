from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import os
import pandas as pd


class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,num_classes)
    def forward(self,x):
        x = self.fc1(x)
        return x
    def plot(self,figure = False,figname = "test",X=None,y=None,model=None):
        """
        X: pd dataframe
        y : pd Series
        """
        
        if(figure==False):
            return
        
        h = 0.02
        thresh = 0.5

        x_train = X.numpy()
        y_train = y.numpy()
        
        x_min, x_max = x_train[:,0].min()- thresh , x_train[:,0].max() + thresh
        y_min,y_max = x_train[:,1].min()- thresh, x_train[:,1].max() + thresh
        # print("1")
        xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
        # print("2")
        fig1,ax =plt.subplots(figsize =(20,5))
        
        fig2,ax = plt.subplots()
        plt.title("Final estimator")

        self.plot_one_estimator(ax,model,xx,yy,X,y,-1) 
        fig1.savefig(os.path.join("plots", str(figname)+"_Fig1.png"))
        fig2.savefig(os.path.join("plots", str(figname)+"_Fig2.png"))
        return fig1,fig2
    
    def plot_one_estimator(self,ax,predictor,xx,yy,X,y,i):
        print("AA")
        if(i==-1):
            X=X.numpy()
            y =y.numpy()
            frame = np.c_[xx.ravel(),yy.ravel()]
            tens = torch.tensor(xx.ravel())
            print("B")
            Z = predictor(tens)
            print("C")

            df = pd.DataFrame(frame,columns=list(self.X.columns))
            Z = predictor(df)
            if(type(Z)!=np.ndarray):
                Z = Z.to_numpy()
            Z = Z.reshape(xx.shape)
            ax.contourf(xx,yy,Z,cmap = plt.cm.RdBu,alpha = 0.7)
            X_ = self.X.to_numpy()
            ax.scatter(X_[:,0],X_[:,1],c= y,cmap=colors.ListedColormap(["#FF0000","#0000FF"]),edgecolors="k")
            ax.set_xlabel("Feature 0")
            ax.set_ylabel("Feature 1")
            ax.set_xlim(xx.min(),xx.max())
            ax.set_ylim(yy.min(),yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
        else:
            frame = np.c_[xx.ravel(),yy.ravel()]
            df = pd.DataFrame(frame,columns=list(self.X.columns))
            sub_df = df[self.column_selected[i]]
            Z = predictor(sub_df)
            if(type(Z)!=np.ndarray):
                Z = Z.to_numpy()
            Z = Z.reshape(xx.shape)
            ax.contourf(xx,yy,Z,cmap = plt.cm.RdBu,alpha = 0.7)
            X_ = self.X.to_numpy()
            ax.scatter(X_[:,0],X_[:,1],c= y,cmap=colors.ListedColormap(["#FF0000","#0000FF"]),edgecolors="k")
            ax.set_xlabel("Feature 0")
            ax.set_ylabel("Feature 1")
            ax.set_xlim(xx.min(),xx.max())
            ax.set_ylim(yy.min(),yy.max())
            ax.set_xticks(())
            ax.set_yticks(())

class DatasetLoader(Dataset):
    def __init__(self,X, y):
        self.X= X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self,index):
        # y is a torch tensor that has the values
        # print(self.X[index],self.y[index])
        return (self.X[index],self.y[index])
    
    
data = load_iris()
X = data['data']
y = data['target']
X = torch.tensor(X)
y = torch.tensor(y)
batch_size = 20
num_epochs = 100
input_size = 4
num_classes = 3
learning_rate = 1e-3
dataset = DatasetLoader(X = X, y=y)
train_set,test_set = torch.utils.data.random_split(dataset,[130,20])
train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset = test_set,batch_size=batch_size,shuffle=True)
model = NN(input_size=input_size,num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr = learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for epoch in range(num_epochs):
    for idx, (data,outputs) in enumerate(train_loader):
        data  =  data.to(device)
        outputs = outputs.to(device).long()
        predictions = model(data.float())
        loss = criterion(predictions,outputs)
        optimizer.zero_grad()
        loss.backward()
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
            _,preds = torch.max(preds,dim=1)
            final_preds = preds
            final_preds = torch.tensor(final_preds)
            no_correct += (final_preds==output).sum()
            no_samples+=len(final_preds)
            accuracy = float(no_correct)/float(no_samples)*100

            print(f"Got {no_correct}/{no_samples} with accuracy {accuracy:.2f}% for {set} set")
    model.train()


# print("came here")
accuracy_check(train_loader,model,"train")         
accuracy_check(test_loader,model,"test")

# model.plot(figure=True,figname="test",X=X,y=y,model=model,)
