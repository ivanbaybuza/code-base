import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np


def adjust_learning_rate(optimizer, learning_rate, epoch, lradj='type1',ratio=0.95):
    if lradj=='type1':
        lr_adjust = {epoch: learning_rate * (ratio ** ((epoch-1) // 1))}
    elif lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def Train_network(model, device, criterion, opt, num_epochs, lr, train_loader, val_loader,accumulation_steps=2):
    
    loss_name = criterion.__class__.__name__
    train_loss = []
    train_accuracy = []
    val_accuracy = []
    model.zero_grad()
    for epoch in range(num_epochs):
        model.train(True) 

        train_accuracy_batch = []

        for batch_no, (X_batch, y_batch) in tqdm(enumerate(train_loader), 
                                                 total=len(train_loader)):
          
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)      

            y_pred_batch = model(X_batch)

            loss = criterion(y_pred_batch, y_batch)
        
            # optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss.append(loss.item())
            train_accuracy_batch.append(loss.item())

                
        train_accuracy_overall = np.mean(train_accuracy_batch)
        train_accuracy.append(train_accuracy_overall.item())


        model.train(False) 
        val_accuracy_batch = []
        for X_batch, y_batch in tqdm(val_loader):
            
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)           
            y_pred_batch = model(X_batch)
            accuracy = criterion(y_pred_batch, y_batch)
            val_accuracy_batch.append(accuracy.item())

        val_accuracy_overall = np.mean(val_accuracy_batch)
        val_accuracy.append(val_accuracy_overall.item())

        adjust_learning_rate(opt, lr, epoch)