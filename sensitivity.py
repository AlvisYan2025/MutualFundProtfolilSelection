#calculate sensitivity for their models 
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from torch.utils.data import Dataset, DataLoader
import statistics
import seaborn as sns
import tensorflow as tf
import pickle
from scipy.stats import pearsonr
import copy
import math
from tqdm import tqdm
import sys 

with open('data.pkl', 'rb') as file:
    all_data = pickle.load(file)
def chronological_sampling_scheme(total_time_periods, split_ratio):
    """
    Simulates the chronological sampling scheme.
    total_time_periods(int):The total number of time periods(469)
    split_ratio (list or array of float): The split ratio [train, val, test]
    """
    split_ratio = np.array(split_ratio)
    if sum(split_ratio) != 1:#normalize
        split_ratio = split_ratio / sum(split_ratio)
    train_size = int(total_time_periods * split_ratio[0])
    val_size = int(total_time_periods * split_ratio[1])
    test_size = total_time_periods - train_size - val_size
    #get indices 
    test_indices = np.arange(0, test_size-1)
    val_indices = np.arange(test_size-1, test_size + val_size-1)#mark
    train_indices = np.arange(test_size + val_size-1, total_time_periods)
    return (test_indices, val_indices, train_indices)
def get_tensor_from_df(df, variables, skip= True):
    #take varibales according to input column names, return a tensor of shape [funds*time, varibles]
    #skip: whether or not to skip rows in which mask is false 
    if skip:
        df = df[df['mask']]
    data = df[variables].to_numpy()
    tensor = torch.tensor(data, dtype=torch.float)
    return tensor
fund_char_names = ['FC2Y', 'Beta', 'OA', 'CF', 'Variance', 'BEME', 'CTO', 'RNA', 'NI','r12_2', 'CF2P', 'r36_13', 'ROA', 'r2_1', 'Resid_Var', 'DPI2A','IdioVol', 'AC', 'PM', 'ATO', 'D2A', 'SUV', 'SGA2S', 'r12_7','PCM', 'LT_Rev', 'D2P', 'Rel2High', 'LTurnover', 'ROE', 'MktBeta','Investment', 'Lev', 'LME', 'E2P', 'ST_Rev', 'Spread', 'Q', 'A2ME','NOA', 'C', 'OP', 'S2P', 'OL', 'AT', 'PROF', 'ages', 'flow','exp_ratio', 'tna', 'turnover', 'Family_TNA', 'fund_no','Family_r12_2', 'Family_flow', 'Family_age', 'F_ST_Rev', 'F_r2_1','F_r12_2']
sentiment = all_data[:,:,-1]
sentiment_list = sentiment.flatten()
sentiment_mean = np.mean(sentiment_list)
def sanitize_input(df, mean=0, mean_macro=sentiment_mean, othernames = fund_char_names, macronames= ['sentiment']):
    #replace missing values in input with mean 
    #for macro variables, using mean_macro
    #other variables need to be bounded between 0.5 and -0.5
    #macro variables need to be bounded between 90 and -90 
    for col in othernames:
        df[col] = df[col].fillna(mean)
        df[col] = np.where((df[col] >= -0.5) & (df[col] <= 0.5), df[col], mean)
    for col in macronames:
        df[col] = df[col].fillna(mean_macro)
        df[col] = np.where((df[col] >= -90) & (df[col] <= 90), df[col], mean_macro)
    return df 
def convert_to_df(data):
    #data -> array of dimension [timesteps, funds, variables]
    #output is a dataframe with columns ['Timestep', 'FundID'] + var_names
    var_names = ['FC2Y', 'Beta', 'OA', 'CF', 'Variance', 'BEME', 'CTO', 'RNA', 'NI',
        'r12_2', 'CF2P', 'r36_13', 'ROA', 'r2_1', 'Resid_Var', 'DPI2A',
        'IdioVol', 'AC', 'PM', 'ATO', 'D2A', 'SUV', 'SGA2S', 'r12_7',
        'PCM', 'LT_Rev', 'D2P', 'Rel2High', 'LTurnover', 'ROE', 'MktBeta',
        'Investment', 'Lev', 'LME', 'E2P', 'ST_Rev', 'Spread', 'Q', 'A2ME',
        'NOA', 'C', 'OP', 'S2P', 'OL', 'AT', 'PROF', 'ages', 'flow',
        'exp_ratio', 'tna', 'turnover', 'Family_TNA', 'fund_no',
        'Family_r12_2', 'Family_flow', 'Family_age', 'F_ST_Rev', 'F_r2_1',
        'F_r12_2', 'sentiment']
    timesteps, funds, variables = data.shape
    timestep_column = np.repeat(np.arange(timesteps), funds)
    fundid_column = np.tile(np.arange(funds), timesteps)
    reshaped_data = data.reshape(timesteps * funds, variables)
    reshaped_data_features = reshaped_data[:,1:]
    reshaped_data_labels = reshaped_data[:,0]
    df = pd.DataFrame(reshaped_data_features, columns=var_names)
    df.insert(0, 'Timestep', timestep_column) 
    df.insert(1, 'FundID', fundid_column)
    df.insert(2, 'label', reshaped_data_labels)
    return df
def add_invalid_label_mask(df, placeholder=-99.99, upper=None, lower=None):
    #df -> dataframe with column 'label'
    #return df with new column 'mask' indicating invalid labels
    #placeholder: value to skip, upper: upper limit, lower: lower limit
    mask = np.array([True]*len(df['label']))
    if placeholder is not None:
        mask_p = (df['label'] != placeholder).to_numpy()
        mask = mask & mask_p
    if upper is not None: 
        mask_u = (df['label'] <= upper).to_numpy()
        mask = mask & mask_u
    if lower is not None: 
        mask_l = (df['label'] >= lower).to_numpy()
        mask = mask & mask_l
    df.insert(len(df.columns), 'mask', mask)
    return df 

class NeuralNet(nn.Module):
    """
    Neural network meta model
    """

    def __init__(self, input_dim, intermediate_dims=(20, 40, 20), dropout=0.9):

        super(NeuralNet, self).__init__()
        self.input_dim = input_dim
        self.intermediate_dims = intermediate_dims
        # define the number of hidden layers
        self.hidden_num = len(intermediate_dims) + 1
        self.dropout = dropout
        self.output_dim = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # define the first hidden layer
        exec("self.hidden_layer1 = nn.Linear({}, {})".format(input_dim, intermediate_dims[0]))
        # define the following hidden layers except for the last layer
        for i in range(len(intermediate_dims) - 1):
            exec(
                "self.hidden_layer{} = nn.Linear({}, {})".format(i + 2, intermediate_dims[i], intermediate_dims[i + 1]))
        # define the last hidden layer
        exec("self.hidden_layer_last = nn.Linear({}, 1)".format(intermediate_dims[-1]))

    def forward(self, x):
        # use loop to determine the next hidden layers
        for i in range(self.hidden_num - 1):
            x = eval("self.hidden_layer{}(x)".format(1 + i))
            x = F.relu(x)
            x = nn.functional.dropout(x, p=self.dropout)

        y = self.hidden_layer_last(x)
        #y = torch.tanh(y)


        return y

    def __repr__(self):
        return "NeuralNet(input_dim={}, output_dim={}, intermediate_dims={}, dropout={})".format(
            self.input_dim.__repr__(), self.output_dim.__repr__(),
            self.intermediate_dims.__repr__(), self.dropout.__repr__()
        )
    def plot(self, train_loss, validation_loss, train_std, val_std, num_epochs, title=''):
        #plot the training graph
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss, label='Train Loss', color='blue')
        if train_std:
            plt.fill_between(epochs, np.array(train_loss) - np.array(train_std), np.array(train_loss) + np.array(train_std), color='blue', alpha=0.2)
        plt.plot(epochs, validation_loss, label='Validation Loss', color='orange')
        if val_std:
            plt.fill_between(epochs, np.array(validation_loss) - np.array(val_std), np.array(validation_loss) + np.array(val_std), color='orange', alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title + " Training and Validation Loss Trends")
        plt.legend()
        last_epoch = num_epochs
        last_train_loss = train_loss[-1]
        last_val_loss = validation_loss[-1]
        plt.annotate(f'Train Loss: {last_train_loss:.2f}', 
                    xy=(last_epoch, last_train_loss), 
                    xytext=(last_epoch, last_train_loss + 0.05),
                    arrowprops=dict(facecolor='blue', shrink=0.05),
                    fontsize=10, color='blue')

        plt.annotate(f'Validation Loss: {last_val_loss:.2f}', 
                    xy=(last_epoch, last_val_loss), 
                    xytext=(last_epoch, last_val_loss + 0.05),
                    arrowprops=dict(facecolor='orange', shrink=0.05),
                    fontsize=10, color='orange')
        #plt.savefig(base+"training_graph_{}.png".format(identifier), dpi=300, bbox_inches='tight')
        plt.show()
    def train_model(self, num_epochs, dataloader_train, dataloader_val, criterion = nn.MSELoss(), learning_rate=0.0025, early_stop=True, regl2=0.001, graph=False):
        '''train model with specified datasets'''
        print('training start')
        print('----------------------')
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        train_loss_avg = [] #average loss over all epochs 
        validation_loss_avg =[] 
        train_loss_std = [] #std of losses over all epochs
        validation_loss_std = []
        lossV= float('inf')
        stop = False
        for epoch in range(num_epochs):
            train_loss = [] #inividual loss for a single epoch
            validation_loss = []
            self.train() 
            running_loss = 0.0
            for inputs, labels in dataloader_train:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                all_params = torch.cat([x.view(-1) for x in self.parameters()])
                l2_regularization = regl2 * torch.norm(all_params, 2)
                loss = loss + l2_regularization
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                running_loss += loss.item()
            lossT = running_loss/len(dataloader_train)
            train_loss_avg.append(lossT)
            #train_loss_std.append(statistics.stdev(train_loss))
            #validation
            self.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in dataloader_val:
                    inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    l2_reg = torch.tensor(0.)
                    for param in self.parameters():
                        l2_reg = l2_reg + torch.norm(param, 2)
                    loss = loss + regl2 * l2_reg
                    validation_loss.append(loss.item())
                    running_val_loss += loss.item()
            if (running_val_loss / len(dataloader_val)>=lossV):
                stop=True
            lossV = running_val_loss / len(dataloader_val)
            validation_loss_avg.append(lossV)
            #validation_loss_std.append(statistics.stdev(validation_loss))
            print(f"Epoch {epoch + 1}/{num_epochs}, Traning Loss: {lossT}, Validation loss: {lossV}")
            if stop and early_stop:
                print('validation stopped converging')
                break 
        if graph:
            self.plot(train_loss_avg, validation_loss_avg, 0, 0, num_epochs)
        return (train_loss_avg, validation_loss_avg, train_loss_std, validation_loss_std)
    def predict(self, inputs):
        #predict at all timestep using model. inputs --> [time, funds, X]
        #return shape [time, funds, 1]
        self.eval()
        self.dropout = 0.0
        with torch.no_grad():
            outputs = self(inputs)
        #np.savetxt('sample.txt', outputs[:, :, -1], fmt='%s')
        #np.savetxt('inputsample.txt', input_features[0,:,:])
        return outputs
    def reinitialize_with_glorot_uniform(self):
        """
        Reinitialize the parameters of the PyTorch model using TensorFlow's 
        default initializer (glorot_uniform, or Xavier uniform).
        """
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init_weights)

##########################################################################calculate sensitivity 
def main(args): 
    identifier = args[1]
    root = sys.argv[2] if len(sys.argv) > 2 else '/scratch/lz2692/equityml_data/'
    def get_out_of_sample_derivative(model, feature, input_timesteps, df): #
        #get outofsample derivative for one model with corresponding feature
        allfeatures = ['ages', 'flow',
            'exp_ratio', 'tna', 'turnover', 'Family_TNA', 'fund_no',
            'Family_r12_2', 'Family_flow', 'Family_age', 'F_ST_Rev', 'F_r2_1',
            'F_r12_2', 'sentiment']
        diff= 0.001
        input_tensor = get_tensor_from_df(df[df['Timestep'].isin(input_timesteps)], allfeatures, False)
        output = model.predict(input_tensor)
        df_plus = df.copy()
        df_plus[feature] = df_plus[feature] + diff
        input_tensor_plus = get_tensor_from_df(df_plus[df_plus['Timestep'].isin(input_timesteps)], allfeatures, False)
        output_plus = model.predict(input_tensor_plus)
        output_diff = (output_plus - output)/diff
        print(output_diff)
        return output_diff

    all_folds = chronological_sampling_scheme(469, [1,1,1])
    list_features = ['ages', 'flow',
            'exp_ratio', 'tna', 'turnover', 'Family_TNA', 'fund_no',
            'Family_r12_2', 'Family_flow', 'Family_age', 'F_ST_Rev', 'F_r2_1',
            'F_r12_2', 'sentiment'] 
    df_features_sensitivity = pd.DataFrame(list_features, columns=['variables'])
    df_features_sensitivity['sensitivity'] = None
    for curr_feature in list_features:
        group_mean = []
        for i in range (1): #i->group number
            df_block  = convert_to_df(all_data)
            df_block = sanitize_input(add_invalid_label_mask(df_block))
            derivative_list = []
            for block in range(3):
                curr_model = torch.load(f'{root}model{block}{identifier}')
                derivative = get_out_of_sample_derivative(curr_model, curr_feature, all_folds[block], df_block)
                derivative_list.append(derivative**2)
            derivative_block = torch.cat(derivative_list, dim=0)
            df_block['derivatives'] = derivative_block.flatten().numpy()
            grouped_df = df_block.groupby('Timestep')['derivatives'].mean().reset_index()
            block_mean = grouped_df['derivatives'].mean()
            group_mean.append(math.sqrt(block_mean))
        df_features_sensitivity.loc[df_features_sensitivity['variables'] == curr_feature, 'sensitivity'] = np.mean(group_mean)
    #normalize across all features
    sensum = df_features_sensitivity['sensitivity'].sum()
    df_features_sensitivity['sensitivity_normalized'] = df_features_sensitivity['sensitivity']/sensum
    #make_sensitivity_graph(df_features_sensitivity)
    df_features_sensitivity.to_csv(f'{root}{identifier}_sensitivity.csv', index=False)
    return df_features_sensitivity

if __name__=='__main__':
    main(sys.argv)
    print(os.system('grep VmPeak /proc/$PPID/status'))