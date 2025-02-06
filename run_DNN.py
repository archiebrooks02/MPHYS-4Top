import numpy as np
import h5py 
from numpy.random import seed
seed_value=420
seed(seed_value)
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import argparse
import yaml
import optuna

class Classifier_MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim1, hidden_dim2, dropout_rate, out_dim):
        super().__init__()
        self.h1 = nn.Linear(in_dim, hidden_dim1)
        self.h2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.h3 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_dim2, out_dim)
        self.out_dim = out_dim        
    
    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.h3(x)
        x = self.out(x)
        return x, F.softmax(x,dim=1)

    def fit(self, train_loader, valid_loader, optimiser, verbose=True):
        _results = []  # Store training and validation metrics
        epochs = load_yaml("dnn_parameters.yaml")["dnn_configuration"]["training"]["epochs"]
        for epoch in range(epochs):
            # Training loop
            self.train()  # Set the model to training mode
            with torch.enable_grad():
                train_loss = 0.0
                for batch, (x_train_batch, y_train_batch) in enumerate(train_loader):
                    self.zero_grad()  # Zero gradients
                    out, prob = self(x_train_batch)  # Forward pass
                    loss = F.cross_entropy(out, y_train_batch)  # Compute loss
                    loss.backward()  # Backpropagation
                    optimiser.step()  # Update weights
                    train_loss += loss.item() * x_train_batch.size(0)

            train_loss /= len(train_loader.dataset)  # Average training loss

            if verbose:
                print("Epoch: {}, Train Loss: {:4f}".format(epoch+1, train_loss))

            # Validation loop
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                correct = 0
                valid_loss = 0.0

                for batch, (x_valid_batch, y_valid_batch) in enumerate(valid_loader):
                    out, prob = self(x_valid_batch)  # Forward pass

                    loss = F.cross_entropy(out, y_valid_batch)  # Compute loss
                    valid_loss += loss.item() * x_valid_batch.size(0)
                    preds = prob.argmax(dim=1, keepdim=True)  # Get predictions
                    correct += preds.eq(y_valid_batch.view_as(preds)).sum().item()  # Count correct

                valid_loss /= len(valid_loader.dataset)  # Average validation loss
                accuracy = correct / len(valid_loader.dataset)  # Calculate accuracy

            if verbose:
                print("Validation Loss: {:4f}, Validation Accuracy: {:4f}".format(valid_loss, accuracy))

            # Store results
            _results.append([epoch, train_loss, valid_loss, accuracy])

        results = np.array(_results)  # Make array of results
        print("Finished Training")
        print("Final validation error: ", 100.0 * (1 - accuracy), "%")
        return results

    def score(self,X_test_tensor,y_test_tensor):
        self.eval()
        with torch.no_grad():
            if isinstance(X_test_tensor, np.ndarray) or isinstance(y_test_tensor, np.ndarray):
                if isinstance(X_test_tensor, np.ndarray):
                    X_test_tensor = torch.from_numpy(X_test_tensor).float()
                if isinstance(y_test_tensor, np.ndarray):
                    y_test_tensor = torch.from_numpy(y_test_tensor).long()
                out, prob = self(X_test_tensor)
                y_pred_NN_partonic = (prob.cpu().detach().numpy().argmax(axis=1))
                return accuracy_score(y_test_tensor,y_pred_NN_partonic)    

def compare_train_test(clf, X_train, y_train, X_test, y_test, xlabel,ax):
    decisions = [] # list to hold decisions of classifier
    for X,y in ((X_train, y_train), (X_test, y_test)): # train and test
        if hasattr(clf, "predict_proba"): # if predict_proba function exists
            d1 = clf.predict_proba(X[y<0.5])[:, 1] # background
            d2 = clf.predict_proba(X[y>0.5])[:, 1] # signal
        else: # predict_proba function doesn't exist
            X_tensor = torch.as_tensor(X, dtype=torch.float) # make tensor from X_test_scaled
            y_tensor = torch.as_tensor(y, dtype=torch.long) # make tensor from y_test
            X_var, y_var = Variable(X_tensor), Variable(y_tensor) # make variables from tensors
            d1 = clf(X_var[y_var<0.5])[1][:, 1].cpu().detach().numpy() # background
            d2 = clf(X_var[y_var>0.5])[1][:, 1].cpu().detach().numpy() # signal
        decisions += [d1, d2] # add to list of classifier decision
    
    highest_decision = max(np.max(d) for d in decisions) # get maximum score
    bin_edges = [] # list to hold bin edges
    bin_edge = -0.1 # start counter for bin_edges
    while bin_edge < highest_decision: # up to highest score
        bin_edge += 0.1 # increment
        bin_edges.append(bin_edge)
    
    ax[1].hist(decisions[0], # background in train set
             bins=bin_edges, # lower and upper range of the bins
             density=True, # area under the histogram will sum to 1
             histtype='stepfilled', # lineplot that's filled
             color='blue', label='Background (train)', # Background (train)
            alpha=0.5 ) # half transparency
    ax[1].hist(decisions[1], # background in train set
             bins=bin_edges, # lower and upper range of the bins
             density=True, # area under the histogram will sum to 1
             histtype='stepfilled', # lineplot that's filled
             color='orange', label='Signal (train)', # Signal (train)
            alpha=0.5 ) # half transparency

    hist_background, bin_edges = np.histogram(decisions[2], # background test
                                              bins=bin_edges, # number of bins in function definition
                                              density=True ) # area under the histogram will sum to 1
    
    scale = len(decisions[2]) / sum(hist_background) # between raw and normalised
    err_background = np.sqrt(hist_background * scale) / scale # error on test background

    width = 0.1 # histogram bin width
    center = (bin_edges[:-1] + bin_edges[1:]) / 2 # bin centres
    
    ax[1].errorbar(x=center, y=hist_background, yerr=err_background, fmt='o', # circles
                 c='blue', label='Background (test)' ) # Background (test)
    
    hist_signal, bin_edges = np.histogram(decisions[3], # siganl test
                                          bins=bin_edges, # number of bins in function definition
                                          density=True ) # area under the histogram will sum to 1
    scale = len(decisions[3]) / sum(hist_signal) # between raw and normalised
    err_signal = np.sqrt(hist_signal * scale) / scale # error on test background
    
    ax[1].errorbar(x=center, y=hist_signal, yerr=err_signal, fmt='o', # circles
                 c='orange', label='Signal (test)' ) # Signal (test)
    
    ax[1].set_xlabel(xlabel) # write x-axis label
    ax[1].set_ylabel("Arbitrary units") # write y-axis label
    ax[1].legend() # add legend

def nn_splitting(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1, random_state=seed_value)

    scaler=StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.as_tensor(X_train_scaled, dtype=torch.float)
    y_train_tensor = torch.as_tensor(y_train, dtype = torch.long)
    X_test_tensor = torch.as_tensor(X_test_scaled, dtype=torch.float)
    y_test_tensor = torch.as_tensor(y_test, dtype = torch.long)

    X_test_variable, y_test_variable = Variable(X_test_tensor), Variable(y_test_tensor)
    X_train_variable, y_train_variable = Variable(X_train_tensor), Variable(y_train_tensor)

    validation_length = int(len(X_train_variable)/10)

    X_valid_variable, y_valid_variable = (X_train_variable[:validation_length],y_train_variable[:validation_length],)
    X_train_nn_variable, y_train_nn_variable = (X_train_variable[validation_length:],y_train_variable[validation_length:],)

    batch_size = 64

    train_data = Data.TensorDataset(X_train_nn_variable,y_train_nn_variable)
    valid_data = Data.TensorDataset(X_valid_variable,y_valid_variable)
    train_loader = Data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
    valid_loader = Data.DataLoader(dataset=valid_data,batch_size=batch_size,shuffle=True)

    return train_loader, valid_loader, X_test_variable, y_test_variable

def NN_train(X,y):

    architecture = load_yaml("dnn_parameters.yaml")["dnn_configuration"]["architecture"]
    input_dim = architecture["input_dim"]
    output_dim = architecture["output_dim"]
    hidden_size_1 = architecture["hidden_1_size"]
    hidden_size_2 = architecture["hidden_2_size"]
    dropout_rate = architecture["dropout_rate"]
    training = load_yaml("dnn_parameters.yaml")["dnn_configuration"]["training"]
    learning_rate = training["learning_rate"]

    NN_clf = Classifier_MLP(in_dim=input_dim, hidden_dim1 = hidden_size_1, hidden_dim2 = hidden_size_2, dropout_rate=dropout_rate, out_dim=output_dim)
    optimiser = torch.optim.Adam(NN_clf.parameters(),lr=learning_rate)

    return NN_clf.fit(X,y,optimiser), NN_clf

def NN_test(X,NN_clf):
    _,prob = NN_clf(X)
    y_pred = (prob.cpu().detach().numpy().argmax(axis=1))
    decisions_nn = (NN_clf(X)[1][:,1].cpu().detach().numpy())

    return y_pred, decisions_nn

def NN_scores(NN_clf,X_train,y_train,X_test,y_test,y_test_variable, decisions_nn):
    fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test_variable, decisions_nn)
    auc_nn = roc_auc_score(y_test_variable, decisions_nn)

    fig,ax = plt.subplots(1,2,figsize=(16,9))

    ax[0].plot(fpr_nn,tpr_nn,linestyle='solid',color='#235789',label=f"NN AUC = {auc_nn:.3f}")
    ax[0].plot([0,1],[0,1],linestyle='dotted',color='grey',label='Luck')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_xlim(0,1)
    ax[0].set_ylim(0,1)
    ax[0].legend(loc='lower right')
    compare_train_test(NN_clf, X_train,y_train,X_test,y_test, "Neural Network Output",ax)
    plt.savefig("NN_outputs.png")
    plt.show()
    print(auc_nn)
    return None

def NN_losses(results):
    epochs = results[:, 0] +1
    train_losses = results[:, 1]
    valid_losses = results[:, 2]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("NN_losses.png")
    plt.show()

def load_data(train_file, test_file):
    with h5py.File("Training_Testing_Split/"+train_file, 'r') as f:
        x_train = f['INPUTS'][:]
        y_train = f['LABELS'][:]

    with h5py.File("Training_Testing_Split/"+test_file, 'r') as f:
        x_test = f['INPUTS'][:]
        y_test = f['LABELS'][:]
    
    return x_train, y_train, x_test, y_test

def load_yaml(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)

def main(train_file,test_file):
    X_train,y_train,X_test,y_test = load_data(train_file, test_file)

    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.as_tensor(X_train_scaled, dtype=torch.float)
    y_train_tensor = torch.as_tensor(y_train, dtype = torch.long)
    X_test_tensor = torch.as_tensor(X_test_scaled, dtype=torch.float)
    y_test_tensor = torch.as_tensor(y_test, dtype = torch.long)
    
    X_test_variable, y_test_variable = Variable(X_test_tensor), Variable(y_test_tensor)
    X_train_variable, y_train_variable = Variable(X_train_tensor), Variable(y_train_tensor)

    validation_length = int(len(X_train_variable)/10)

    X_valid_variable, y_valid_variable = (X_train_variable[:validation_length],y_train_variable[:validation_length],)
    X_train_nn_variable, y_train_nn_variable = (X_train_variable[validation_length:],y_train_variable[validation_length:],)

    batch_size = load_yaml("dnn_parameters.yaml")["dnn_configuration"]["training"]["batch_size"]

    train_data = Data.TensorDataset(X_train_nn_variable,y_train_nn_variable)
    valid_data = Data.TensorDataset(X_valid_variable,y_valid_variable)
    train_loader = Data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
    valid_loader = Data.DataLoader(dataset=valid_data,batch_size=batch_size,shuffle=True)

    print("Training neural network...")
    results,nn_clf = NN_train(train_loader,valid_loader)
    print("Training complete.")

    print("Testing neural network...")
    y_pred,decisions_nn = NN_test(X_test_variable,nn_clf)
    print("Testing complete.")

    NN_scores(nn_clf,X_train_scaled,y_train,X_test_scaled,y_test,y_test_variable, decisions_nn)
    NN_losses(results)

def objective(trial):
    X_train,y_train,X_test,y_test = load_data("train.h5", "test.h5")
    train_loader, valid_loader, X_test_variable, y_test_variable = nn_splitting(X_train,y_train)
    architecture = load_yaml("dnn_parameters.yaml")["dnn_configuration"]["architecture"]
    input_dim = architecture["input_dim"]
    output_dim = architecture["output_dim"]
    hidden_size_1 = trial.suggest_int('hidden_size_1', 1, 100)
    hidden_size_2 = trial.suggest_int('hidden_size_2', 1, 100)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    NN_clf = Classifier_MLP(in_dim=input_dim, hidden_dim1 = hidden_size_1, hidden_dim2 = hidden_size_2, dropout_rate=dropout_rate, out_dim=output_dim)
    optimiser = torch.optim.Adam(NN_clf.parameters(),lr=learning_rate)

    decisions_nn = (NN_clf(X_test_variable)[1][:,1].cpu().detach().numpy())
    auc_nn = roc_auc_score(y_test_variable, decisions_nn)
    return auc_nn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DNN and plot training loss.")
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training data file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the testing data file')
    
    args = parser.parse_args()
    main(args.train_file, args.test_file)