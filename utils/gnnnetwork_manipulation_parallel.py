import numpy as np
import os
import torch
import torch.nn as nn

#https://www.zhihu.com/search?type=content&q=SyncBatchNorm       
def reduce_value(value, average=True):
    world_size = torch.distributed.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value
    with torch.no_grad():
        torch.distributed.all_reduce(value)   # 对不同设备之间的value求和
        if average:  # 如果需要求平均，获得多块GPU计算loss的均值
            value /= world_size
    return value        
 
def train_gcn(log_interval, model, device, train_loader, optimizer, criterion, epoch, num_epochs, scaler, grad_clip, model_name):
    model.train()

    n_total_steps = len(train_loader)
    losses = []
    epoch_loss = 0.

    for i, data in enumerate(train_loader):
        data = data.to(device)
        if model_name == 'MoNet':
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        elif model_name == "GravNet":
            out = model(data.x, data.batch)
        loss = criterion(out, data.y)
        optimizer.zero_grad()
        loss.backward()
            
        if grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        optimizer.step()
        loss = reduce_value(loss, average=True)
        losses.append(loss.item())

        if (i+1) % log_interval  == 0:
            #print(data.x.shape)
            #print(data.batch.shape)
            #print(out.shape)
            #print(data.y.shape)
            print (f'Train Epoch [{epoch+1}/{num_epochs}], batch {i+1} in {n_total_steps},  loss: {loss.item():.5f}.')

    epoch_loss = (np.mean(losses)).tolist()
    return epoch_loss



def validation_gcn(model, device, optimizer, criterion, valid_loader, model_name):
    model.eval()

    total_loss, total_accuracy, total_samples = 0, 0, 0

    with torch.no_grad():
        valid_loss, valid_acc = [], []
        epoch_loss = 0.
        #corr, total = 0., 0.

        for i, data in enumerate(valid_loader):
            data = data.to(device)
            if model_name == 'MoNet':
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            elif model_name == "GravNet":
                out = model(data.x, data.batch)

            loss = criterion(out, data.y)
            loss = reduce_value(loss, average=True)
            valid_loss.append(loss.item())

            # calculate accuracy
            #_, predicted = torch.max(out, 1)
            #corr += (predicted == labels).sum().item()
            #total+= data.x.size(0)
            #acc0 = corr / total

    epoch_loss = (np.mean(valid_loss)).tolist()
    return epoch_loss



def test_accuracy_gcn(model, device, test_loader):
    # Try to test accuracy on a single GPU
    model.eval()

    correct = 0
    score = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)

            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = out.argmax(dim=1)
            corr = int((pred == data.y).sum())
            correct += corr
            print(f'Batch {i} in test_loader has accuracy = {float(corr)/data.y.size(0)}.')

        return correct / len(test_loader.dataset)













