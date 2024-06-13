#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Carlos Perez De Jesus

###############################################################################
                        IRMAE-WD-B Code
###############################################################################

This script demos IRMAE-WB applied to the dataset Kuramoto-Sivashinsky equation
(KSE), L=22 with a weight decay value of 10**-6

###############################################################################

"""



import numpy as np
import matplotlib.pyplot as plt
import pickle as p
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torch as T
from torch.utils.data import DataLoader

class irmae_wd_b(nn.Module):
    def __init__(self, ambient_dim=64, code_dim=20, filepath='testae'):
        super(irmae_wd_b, self).__init__()
        
        self.ambient_dim = ambient_dim
        self.code_dim = code_dim
        
        
        self.lin_enc_1 = nn.Linear(self.ambient_dim, 512)
        self.nl_1      = nn.ReLU()
        self.lin_enc_2 = nn.Linear(512,256)
        self.nl_2      = nn.ReLU()
        self.lin_enc_3 = nn.Linear(256,self.code_dim)
        
        self.lin1      = nn.Linear(self.code_dim, self.code_dim, bias=False)
        self.lin2      = nn.Linear(self.code_dim, self.code_dim, bias=False)
        self.lin3      = nn.Linear(self.code_dim, self.code_dim, bias=False)
        self.lin4      = nn.Linear(self.code_dim, self.code_dim, bias=False)
        self.lin5      = nn.Linear(self.code_dim, self.code_dim, bias=False)
        self.lin6      = nn.Linear(self.code_dim, self.code_dim, bias=False)
        self.lin7      = nn.Linear(self.code_dim, self.code_dim, bias=False)
        self.lin8      = nn.Linear(self.code_dim, self.code_dim, bias=False)
        self.lin9      = nn.Linear(self.code_dim, self.code_dim, bias=False)
        self.lin10     = nn.Linear(self.code_dim, self.code_dim, bias=False)
        
        self.decoder1 = nn.Sequential(
            nn.Linear(self.code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.ambient_dim))
        
        self.decoder2 = nn.Sequential(
            nn.Linear(self.code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.ambient_dim))
        
        self.decoder3 = nn.Sequential(
            nn.Linear(self.code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.ambient_dim))
        
        self.decoder4 = nn.Sequential(
            nn.Linear(self.code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.ambient_dim))
        
        self.decoder5 = nn.Sequential(
            nn.Linear(self.code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.ambient_dim))
        
        self.decoder6 = nn.Sequential(
            nn.Linear(self.code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.ambient_dim))
        
        self.decoder7 = nn.Sequential(
            nn.Linear(self.code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.ambient_dim))
        
        self.decoder8 = nn.Sequential(
            nn.Linear(self.code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.ambient_dim))
        
        self.decoder9 = nn.Sequential(
            nn.Linear(self.code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.ambient_dim))
        
        self.decoder10 = nn.Sequential(
            nn.Linear(self.code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.ambient_dim))
        
    def forward(self,x):
        code = self.lin_enc_1(x) 
        code = self.nl_1(code)
        code = self.lin_enc_2(code)
        code = self.nl_2(code)
        code = self.lin_enc_3(code)
        
        code_l1 = self.lin1(code)
        code_l2 = self.lin2(code_l1)
        code_l3 = self.lin3(code_l2)
        code_l4 = self.lin4(code_l3)
        code_l5 = self.lin5(code_l4)
        code_l6 = self.lin6(code_l5)
        code_l7 = self.lin7(code_l6)
        code_l8 = self.lin8(code_l7)
        code_l9 = self.lin9(code_l8)
        code_l10 = self.lin10(code_l9)
        
        
        xhat_l1 = self.decoder1(code_l1)
        xhat_l2 = self.decoder2(code_l2)
        xhat_l3 = self.decoder3(code_l3)
        xhat_l4 = self.decoder4(code_l4)
        xhat_l5 = self.decoder5(code_l5)
        xhat_l6 = self.decoder6(code_l6)
        xhat_l7 = self.decoder7(code_l7)
        xhat_l8 = self.decoder8(code_l8)
        xhat_l9 = self.decoder9(code_l9)
        xhat_l10 = self.decoder10(code_l10)
        
        return [xhat_l1,xhat_l2,xhat_l3,xhat_l4,xhat_l5,xhat_l6,xhat_l7,xhat_l8,xhat_l9,xhat_l10]
    
    def encode(self, x): 
        code = self.lin_enc_1(x) 
        code = self.nl_1(code)
        code = self.lin_enc_2(code)
        code = self.nl_2(code)
        code = self.lin_enc_3(code)
        
        code_l1 = self.lin1(code)
        code_l2 = self.lin2(code_l1)
        code_l3 = self.lin3(code_l2)
        code_l4 = self.lin4(code_l3)
        code_l5 = self.lin5(code_l4)
        code_l6 = self.lin6(code_l5)
        code_l7 = self.lin7(code_l6)
        code_l8 = self.lin8(code_l7)
        code_l9 = self.lin9(code_l8)
        code_l10 = self.lin10(code_l9)
        
        return [code_l1,code_l2,code_l3,code_l4,code_l5,code_l6,code_l7,code_l8,code_l9,code_l10]

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))    

device = T.device("cuda" if T.cuda.is_available() else "cpu")

# For outputting info when running on compute nodes
def output(text):
    # Output epoch and Loss
    name='Out.txt'
    newfile=open(name,'a+')
    newfile.write(text)
    newfile.close()


# Get Covariance, SVD
def getSVD(code_data):
    #Compute covariance matrix and singular values
    code_mean = code_data.mean(axis=0)
    code_std = code_data.std(axis=0)
    code_data = (code_data - code_mean)
    
    covMatrix = (code_data.T @ code_data) / len(dataset)
    u, s, v = np.linalg.svd(covMatrix, full_matrices=True)

    return code_mean, code_std, covMatrix, u, s, v


if __name__ == '__main__':
    
    #Parameters
    num_epochs = 200
    batch_size = 128
    learning_rate = 1e-3
    train_frac = 0.8
    train = True

    #Weight decay parameter
    wd = 6
    wd_param = 10**-wd
    

    #Initialize model, define loss function, set optimizer
    model = irmae_wd_b().to(device)
    model.double()
    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd_param)

    #Load data for Kuramoto-Sivashinsky equation (KSE), L=22
    matdata = scipy.io.loadmat('./KSE_L22.mat')

    rawdata = matdata['ut']
    rawdata = rawdata[0:64,100:40000]
    rawdata = rawdata.T
    mean = rawdata.mean(axis=0)
    std = rawdata.std(axis=0)


    #Normalize Data with mean and std
    dataset = (rawdata - mean)/std
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    training_dataset = rawdata[0:int(len(rawdata)*train_frac),:]
    training_dataset_small = np.copy(dataset[0:500,:])
    training_dataset_T = T.tensor(training_dataset_small, dtype=T.double).to(device)
    dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    #Data for Computing SVD
    chunk = np.copy(dataset)
    Testdata = T.tensor(chunk, dtype=T.double).to(device)

    #Initialize Storage Matrices
    tot_err = []
    cov_save = np.array([])
    s_save = np.array([])
    code_id_list = []
    code_id_list_epoch1 = []


    if train:
        for epoch in range(num_epochs):
            
            if epoch <= num_epochs - int(num_epochs*0.1):
                
                reconstructed_for_index = model(training_dataset_T)
                loss_l1 = loss_function(reconstructed_for_index[0], training_dataset_T)
                loss_l2 = loss_function(reconstructed_for_index[1], training_dataset_T)
                loss_l3 = loss_function(reconstructed_for_index[2], training_dataset_T)
                loss_l4 = loss_function(reconstructed_for_index[3], training_dataset_T)
                loss_l5 = loss_function(reconstructed_for_index[4], training_dataset_T)
                loss_l6 = loss_function(reconstructed_for_index[5], training_dataset_T)
                loss_l7 = loss_function(reconstructed_for_index[6], training_dataset_T)
                loss_l8 = loss_function(reconstructed_for_index[7], training_dataset_T)
                loss_l9 = loss_function(reconstructed_for_index[8], training_dataset_T)
                loss_l10 = loss_function(reconstructed_for_index[9], training_dataset_T)
                
                w_1 = np.exp(-loss_l1.item())
                w_2 = np.exp(-loss_l2.item())
                w_3 = np.exp(-loss_l3.item())
                w_4 = np.exp(-loss_l4.item())
                w_5 = np.exp(-loss_l5.item())
                w_6 = np.exp(-loss_l6.item())
                w_7 = np.exp(-loss_l7.item())
                w_8 = np.exp(-loss_l8.item())
                w_9 = np.exp(-loss_l9.item())
                w_10 = np.exp(-loss_l10.item())
                
                w_sum = w_1+w_2+w_3+w_4+w_5+w_6+w_7+w_8+w_9+w_10
                
                what_1 = w_1/w_sum
                what_2 = w_2/w_sum
                what_3 = w_3/w_sum
                what_4 = w_4/w_sum
                what_5 = w_5/w_sum
                what_6 = w_6/w_sum
                what_7 = w_7/w_sum
                what_8 = w_8/w_sum
                what_9 = w_9/w_sum
                what_10 = w_10/w_sum
                
                what_1_range = [0,what_1]
                what_2_range = [what_1_range[1],what_1_range[1]+what_2]
                what_3_range = [what_2_range[1],what_2_range[1]+what_3]
                what_4_range = [what_3_range[1],what_3_range[1]+what_4]
                what_5_range = [what_4_range[1],what_4_range[1]+what_5]
                what_6_range = [what_5_range[1],what_5_range[1]+what_6]
                what_7_range = [what_6_range[1],what_6_range[1]+what_7]
                what_8_range = [what_7_range[1],what_7_range[1]+what_8]
                what_9_range = [what_8_range[1],what_8_range[1]+what_9]
                what_10_range = [what_9_range[1],what_9_range[1]+what_10]
                
                randnum = np.random.uniform(0,1)
                
                check_L1 = randnum>=what_1_range[0] and randnum<what_1_range[1]
                check_L2 = randnum>=what_2_range[0] and randnum<what_2_range[1]
                check_L3 = randnum>=what_3_range[0] and randnum<what_3_range[1]
                check_L4 = randnum>=what_4_range[0] and randnum<what_4_range[1]
                check_L5 = randnum>=what_5_range[0] and randnum<what_5_range[1]
                check_L6 = randnum>=what_6_range[0] and randnum<what_6_range[1]
                check_L7 = randnum>=what_7_range[0] and randnum<what_7_range[1]
                check_L8 = randnum>=what_8_range[0] and randnum<what_8_range[1]
                check_L9 = randnum>=what_9_range[0] and randnum<what_9_range[1]
                check_L10 = randnum>=what_10_range[0] and randnum<what_10_range[1]
                
                check_all = [check_L1,check_L2,check_L3,check_L4,check_L5,check_L6,check_L7,check_L8,check_L9,check_L10]
                
                code_id = check_all.index(True)
                
                for snapshots in dataloader:
                    inputs = snapshots
                    inputs = inputs.to(device)
                    
                    #Forward pass, compute loss
                    reconstructed = model(inputs)
                    loss = loss_function(reconstructed[code_id], inputs)
                    
                    #Back prop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # ===================log========================
                print('epoch [{}/{}], loss:{:.6f}'
                    .format(epoch + 1, num_epochs, loss.item()))
                tot_err.append(loss.item())
                code_id_list.append(code_id)

                
                if epoch % 10 == 0:
                    output('Iter {:04d} | Total Loss {:.6f} '.format(epoch, loss.item())+'\n')
                    code_data = model.encode(Testdata)
                    code_data = code_data[code_id]
                    code_data = code_data.detach().numpy()
                    _, _, temp_cov, _, temp_s, _ = getSVD(code_data)
                    s_save = np.hstack([s_save, temp_s[:,np.newaxis]]) if s_save.size else temp_s[:,np.newaxis]
                    
            else:
                reconstructed_for_index = model(training_dataset_T)
                loss_l1 = loss_function(reconstructed_for_index[0], training_dataset_T)
                loss_l2 = loss_function(reconstructed_for_index[1], training_dataset_T)
                loss_l3 = loss_function(reconstructed_for_index[2], training_dataset_T)
                loss_l4 = loss_function(reconstructed_for_index[3], training_dataset_T)
                loss_l5 = loss_function(reconstructed_for_index[4], training_dataset_T)
                loss_l6 = loss_function(reconstructed_for_index[5], training_dataset_T)
                loss_l7 = loss_function(reconstructed_for_index[6], training_dataset_T)
                loss_l8 = loss_function(reconstructed_for_index[7], training_dataset_T)
                loss_l9 = loss_function(reconstructed_for_index[8], training_dataset_T)
                loss_l10 = loss_function(reconstructed_for_index[9], training_dataset_T)
                loss_list = [loss_l1,loss_l2,loss_l3,loss_l4,loss_l5,loss_l6,loss_l7,loss_l8,loss_l9,loss_l10]
                code_id = loss_list.index(min(loss_list))
                
                for snapshots in dataloader:
                    inputs = snapshots
                    inputs = inputs.to(device)
                    
                    #Forward pass, compute loss
                    reconstructed = model(inputs)
                    loss = loss_function(reconstructed[code_id], inputs)
                    
                    #Back prop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # ===================log========================
                print('epoch [{}/{}], loss:{:.6f}'
                    .format(epoch + 1, num_epochs, loss.item()))
                tot_err.append(loss.item())
                code_id_list.append(code_id)
                # Print out the Loss and the time the computation took
                if epoch % 10 == 0:
                    output('Iter {:04d} | Total Loss {:.6f} '.format(epoch, loss.item())+'\n')
                    code_data = model.encode(Testdata)
                    code_data = code_data[code_id]
                    code_data = code_data.detach().numpy()
                    _, _, temp_cov, _, temp_s, _ = getSVD(code_data)
                    s_save = np.hstack([s_save, temp_s[:,np.newaxis]]) if s_save.size else temp_s[:,np.newaxis]


        T.save(model.state_dict(), 'IRMAEWDB_AE.pt')
        p.dump(tot_err,open('err.p','wb'))
                #Print Training Curve
        fig = plt.figure(num=None, figsize=(7, 7), dpi=100, facecolor='w', edgecolor='w')
        plt.semilogy(tot_err,c='k', label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('TrainCurve.png')

        
    else:
        model.load_state_dict(T.load('IRMAEWDB_AE.pt'))
        print('testing')
        
        
    #Get data for plotting and computing singular values/basis vectors
    code_data = model.encode(Testdata)
    code_data = code_data[code_id]
    code_data = code_data.detach().numpy()
    code_mean, code_std, covMatrix, u, s, v = getSVD(code_data)

    #Save Results
    p.dump([code_mean,code_std],open('code_musigma.p','wb'))
    p.dump([u,s,v],open('code_svd.p','wb'))
    p.dump(s_save,open('training_svd.p','wb'))
    p.dump(code_id_list,open('code_id_list.p','wb'))
    
    #Plotting singular values
    fig = plt.figure(num=None, figsize=(7, 7), dpi=100, facecolor='w', edgecolor='w')
    plt.semilogy(s/s[0],'ko--')
    plt.xlabel('Singular value')
    plt.ylabel('Singular value of covariance of z')
    plt.tight_layout()
    plt.savefig(open('code_sValues.png','wb'))
        
        
        
    
