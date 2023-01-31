import os, time, argparse, matplotlib, sklearn, torch, torchmetrics, shap
matplotlib.use('Agg') # Fix for $> _tkinter.TclError: couldn't connect to display "localhost:36.0"
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics import PrecisionRecallCurve, Accuracy, Precision, Recall, ROC
from torchviz import make_dot

# Simple customised dataset class inhertis from Dataset class which is accepted by torch models
# Takes dataframe and returns tensors of input and labels
class custom_train_dataset(Dataset):
    def __init__(self, df, input_columns, target_column, transform=None, target_transform=None):
        self.df = df
        # make tensors for values of inputs and relveant labels
        source_combs = self.df[input_columns].values
        target_labels = self.df[target_column].values
        self.x_train = torch.tensor(source_combs,dtype=torch.float32)
        self.y_train = torch.tensor(target_labels,dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x_ = self.x_train[index]
        y_ = self.y_train[index]

        if self.transform:
            x_ = self.transform(x_)

        return x_, y_

# Make subclass of torch nn module to extend modules functionality
class NeuralNetwork(nn.Module):
    def __init__(self,nvars):
        # return a temporary object of superclass so we can call superclass' methods
        super(NeuralNetwork, self).__init__()
        # Initialise layers
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(), #Flattens contiguous range of dimensions into a tensor
            nn.Linear(nvars,24),
            nn.ReLU(),
            nn.Linear(24,12),
            nn.ReLU(),
            nn.Linear(12,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4,1),
        )
    # Method to implement operations on input data
    # Passing input data to model automatically executes models forward method
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set (averaged batch loss * batch size) to 0
    batch_loss = 0.0
    # For each batch in the dataloader
    for batch, (X,y) in enumerate(dataloader):
        # Set any learned gradients for optimised tensors to zero (as they will otherwise be from previous batch using different parameters)
        optimizer.zero_grad()
        # Alter dimensions of labels tensor
        y = y.unsqueeze(1)
        # Compute prediction and loss for the whole batch
        logits = model(X)
        # Loss computation for the batch
        loss = loss_fn(logits, y)
        ### Backpropagation ###
        # compute sum of gradients of given tensor for this batch
        loss.backward()
        # Update weights
        optimizer.step()
        # Add loss to batch loss
        batch_loss += loss.item() * X.size(0)
        # Report every X batches
        if batch % 10 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f'loss:{loss:.4} [{current}/{size}]')
    # Return: averaged batch loss * batch size , model
    return batch_loss, model

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_loss = 0.0
    # gradient calculation disabled (not needed in evaluation)
    with torch.no_grad():
        # for each batch
        for X,y in dataloader:
            # make prediction using model
            logits = model(X)
            y = y.unsqueeze(1) # required to ensure same dimensions
            # add loss for example to batch loss using prediction and ground truth
            test_loss = loss_fn(logits, y)
            # Add loss to batch loss
            batch_loss += test_loss.item() * X.size(0)
            # Report every X batches
            #if batch % 10 == 0:
            #    loss, current = test_loss.item(), batch*len(X)
            #    print(f'test_loss:{test_loss:.4} [{current}/{size}]')

        return batch_loss

def eval_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0
    # gradient calculation disabled (not needed in evaluation)
    with torch.no_grad():
        #pr_curve = PrecisionRecallCurve(pos_label=1)
        #roc = ROC(pos_label=1)
        all_truth_labels = []
        all_predictions = []

        # for each batch
        for X,y in dataloader:
            # make prediction using model
            logits = model(X)
            y = y.unsqueeze(1) # required to ensure same dimensions
            all_truth_labels += y.tolist()
            all_predictions += torch.sigmoid(logits).tolist()
            # add loss for example to batch loss using prediction and ground truth
            test_loss += loss_fn(logits, y).item()
            # pair predictions with data truth
            test_stack = torch.stack((torch.sigmoid(logits), y), dim=1)

        #print('New event\n',list(zip(all_predictions,all_truth_labels)))
        # Look for combination with highest prediciton score
        max_pred_ = max(all_predictions)
        # get the index of this combination
        max_pred_index_ = all_predictions.index(max_pred_)
        # Look for combination with highest truth label (i.e. == 1)
        max_truth_ = max(all_truth_labels)
        # Get the index
        max_truth_index_ = all_truth_labels.index(max_truth_)
        #print( 'predicted index: %s, truth index: %s' % (max_pred_index_, max_truth_index_) )
        return max_pred_index_, max_truth_index_

def pT_rank_sel(test_dataset):
    sum_pt_ = -100
    for row in test_dataset.index:
        if test_dataset['bmatched_jet_pt'][row] + test_dataset['lmatched_jet_pt'][row] > sum_pt_:
            sum_pt_ = test_dataset['bmatched_jet_pt'][row] + test_dataset['lmatched_jet_pt'][row]
            label_pT_high = test_dataset['label'][row]

    return label_pT_high

def main():
    t0 = time.time()

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-t', '--newtrain', dest='newtraining', help= 'Training a new model?', default='0', type=int)
    args = parser.parse_args()
    newtraining = args.newtraining
    # Input Variables
    infile_name_ = '../truth_reco/dataframes/test_frames/100k_result_drvar.parquet.gzip'
    indf = pd.read_parquet(infile_name_)
    input_columns_ = [
    'bmatched_jet_pt','bmatched_jet_eta','bmatched_jet_phi','bmatched_jet_mass',
    'lmatched_jet_pt','lmatched_jet_eta','lmatched_jet_phi','lmatched_jet_mass',
    'dR_bmatched_lmatched_jets','dR_bmatched_jet_lep1','dR_bmatched_jet_lep2','dR_lmatched_jet_lep1','dR_lmatched_jet_lep2',
    'invmass_bjlj',
    'lep1_pt','lep1_eta','lep1_phi','lep1_mass',
    'lep2_pt','lep2_eta','lep2_phi','lep2_mass',
    'jet3_pt','jet3_eta','jet3_phi','jet3_mass',
    'jet4_pt','jet4_eta','jet4_phi','jet4_mass'
    ]

    # Hyperparameters
    learning_rate = 0.00001
    batchsize = 500
    n_epochs = 100

    # Choose device to run on
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # Create datasets
    train_df, test_df = train_test_split(indf, test_size=0.2, shuffle=False)
    test_df, val_df = train_test_split(test_df, test_size=0.5, shuffle=False)

    train_dataset = custom_train_dataset(train_df, input_columns_, 'label')
    test_dataset = custom_train_dataset(test_df, input_columns_, 'label')

    # DataLoaders for train datasets
    training_data_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    testing_data_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    # Test dataLoaders by loading batches of features and labels
    train_features, train_labels = next(iter(training_data_loader))
    print(f"Features batch shape: {train_features.size()} ")
    print(f"Labels batch shape: {train_labels.size()} ")

    # Class weights
    # Be careful using class weights:
    #       - Too large and they can cause instability in the training such that the algorithm doesnt learn
    #       - Too small and if you have class imbalance, class(es) can be ignored to obtain high accuracy
    n_sig = [x for x in train_labels if x == 1]
    n_bkg = [x for x in train_labels if x == 0]
    class_ratio = len(n_bkg)/len(n_sig)
    print('sig/bkg class ratio: ', class_ratio)

    # Define loss function
    #loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_ratio))
    loss_function = nn.BCEWithLogitsLoss()

    if newtraining == 1:
        # Generate model and move to device
        model = NeuralNetwork(len(input_columns_)).to(device)
        print(model)

        # Initialsed optimiser with models parameters
        # n.b. must call parameters method on model when initialising optimiser
        optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)

        # Looping over entire dataset
        train_losses = []
        test_losses = []
        iters = 0
        for epoch in range(n_epochs):
            print(f'Epoch {epoch+1}\n-----------------------')
            loss_ , model_ = train_loop(training_data_loader, model, loss_function, optimizer)
            test_loss_ = test_loop(testing_data_loader, model, loss_function)
            # Calculate the average loss over the epoch
            av_training_loss = loss_/len(training_data_loader.sampler)
            av_testing_loss = test_loss_/len(testing_data_loader.sampler)
            print('Epoch av. training loss: ',  av_training_loss)
            print('Epoch av. testing  loss: ',  av_testing_loss)
            train_losses.append(av_training_loss)
            test_losses.append(av_testing_loss)

        model_outdir_name = 'model_'+str(learning_rate)+'_'+str(batchsize)+'_'+str(n_epochs)
        if not os.path.isdir(model_outdir_name):
            print('Making model directory: ', model_outdir_name)
            os.makedirs(model_outdir_name)
        model_save_ = os.path.join(model_outdir_name,'saved_model.pt')
        print('Saving model @: ', model_save_)
        torch.save(model_.state_dict(), model_save_)

        # Training monitoring plots
        plt.figure()
        plt.plot(train_losses, label='training')
        plt.plot(test_losses, label='testing')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.title(f'lr:{learning_rate:4f}')
        plt.savefig(os.path.join(model_outdir_name,f'loss_vs_epoch_{learning_rate}.png'))
        plt.clf()
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = NeuralNetwork(len(input_columns_))
        load_model_ = os.path.join(model_outdir_name,'saved_model.pt')
        print('loading model: ', load_model_)
        model.load_state_dict(torch.load(load_model_))
        model.to(device)


    # N.B. for the Evaluation loop, we pass data per 'Entry'
    # >>>> checking all combinations for one event, select highest score, check label
    print('Evaluation')
    groups_ = val_df.groupby('Entry')
    num_corr_ptSel = 0
    num_corr = 0
    total_ = 0
    for evnum in groups_.groups.keys():
        val_dataset = custom_train_dataset(groups_.get_group(evnum), input_columns_, 'label')
        val_data_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        max_pred_idx , corr_idx_ = eval_loop(val_data_loader, model, loss_function)
        label_high_pt_sel = pT_rank_sel(groups_.get_group(evnum))

        total_ += 1
        if label_high_pt_sel == 1:
            num_corr_ptSel+=1
        if max_pred_idx == corr_idx_:
            num_corr+=1

    # visualise NN architecture
    tmp_data = custom_train_dataset(test_df, input_columns_, 'label')
    tmp_dataloader = DataLoader(tmp_data, batch_size=len(tmp_data), shuffle=False)
    xtmp,ytmp = next(iter(tmp_dataloader))
    ypred_tmp = model(xtmp)
    arch_im = make_dot(ypred_tmp.mean(), params=dict(model.named_parameters())).render(os.path.join(model_outdir_name,"DNN_torchviz"), format="png")

    # variable ranking using shaply values
    shapX_ = xtmp[:1000]
    dexplainer = shap.DeepExplainer(model, shapX_)
    dex_shap_values_ = dexplainer.shap_values(shapX_)
    shap.summary_plot(dex_shap_values_, shapX_, input_columns_)
    plt.savefig(os.path.join(model_outdir_name,'shap_vals.png'))
    plt.clf()

    # Proportion of events the algorithm/pTrank correctly guessed the correct combination
    percent_corr_ptSel = (num_corr_ptSel/total_)*100
    percent_corr = (num_corr/total_)*100
    print(f'{num_corr}/{total_} ({percent_corr:.3} percent) events correctly assigned by algo')
    print(f'{num_corr_ptSel}/{total_} ({percent_corr_ptSel:.3} percent) events correctly assigned by selecting high-pt pair')
    with open(os.path.join(model_outdir_name,'results.txt'), 'w') as f:
        lines =[f'{num_corr}/{total_} ({percent_corr:.3} percent) events correctly assigned by algo', f'{num_corr_ptSel}/{total_} ({percent_corr_ptSel:.3} percent) events correctly assigned by selecting high-pt pair']
        f.write('\n'.join(lines))
    print('FIN!')
    return


if __name__ == '__main__':
    main()
