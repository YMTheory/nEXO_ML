import numpy as np

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
 
        
def train(log_interval, model, device, train_loader, optimizer, criterion, epoch, num_epochs,  scaler, grad_clip=0):
#def train(log_interval, model, device, train_loader, optimizer, criterion, epoch, num_epochs, lr_sched, scaler, grad_clip=0):
    # set model as training mode
    # https://stackoverflow.com/questions/53879727/pytorch-how-to-deactivate-dropout-in-evaluation-mode
    model.train()

    n_total_steps = len(train_loader) # How many batches the data_loader is separated.
    losses = []
    epoch_loss = 0
    corr, total = 0, 0
    acc = []
    epoch_acc = 0

    lrs = []

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
                                
        # Forward pass
        outputs = model(images) # output here is 2D array (scores for each class) with length = batch_size.
        loss = criterion(outputs, labels) # loss here is 1 digit calculated from all batch data.
        #print(f'Loss for batch {i} is {loss}.')
         
        # Backward and optimize        
        optimizer.zero_grad()        
        loss.backward()  
              
        # Gradient clipping
        if grad_clip: 
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)      

        optimizer.step()
        
        loss = reduce_value(loss, average=True)  #  在单GPU中不起作用，多GPU时，获得所有GPU的loss的均值。
        losses.append(loss.item())

        #_, predicted = outputs.max(1)
        #total += labels.size(0)
        #corr += predicted.eq(labels).sum().item()
        #acc0 = corr / total
        #correct = torch.tensor(acc0, dtype=torch.float32).to('cuda').to_dense()
        #correct = reduce_value(correct, average=True)
        #acc.append(correct.item())
        
        lrs.append( optimizer.param_groups[0]["lr"] )
        #lr_sched.step()        
        
        
        if (i+1) % log_interval  == 0:
            print (f'Train Epoch [{epoch+1}/{num_epochs}], batch {i+1} in {n_total_steps}, LR={lrs[-1]:.2E},  loss: {loss.item():.5f}.')
            #print (f'Train Epoch [{epoch+1}/{num_epochs}], batch {i+1} in {n_total_steps}, LR={lrs[-1]:.2E},  loss: {loss.item():.5f}, ==> correct: {corr}, total: {total}, accuracy: {correct.item():.5f}')
            # 此时返回的loss和accuracy都是平均了所有进程的结果，是当前这一个batch里的样本计算出来的结果。
            

    epoch_loss =(np.mean(losses)).tolist()    
    #epoch_acc = (np.mean(acc)).tolist() # mean for all processes and until current epoch
    # 这里返回的是所有batch的loss, accuracy平均的结果。
    return epoch_loss, lrs
    #return epoch_loss, epoch_acc, lrs




def validation(model, device, optimizer, criterion, test_loader):
    # set model as testing mode
    # https://stackoverflow.com/questions/53879727/pytorch-how-to-deactivate-dropout-in-evaluation-mode
    model.eval()
    
    total_loss, total_accuracy, total_samples = 0, 0, 0

    with torch.no_grad():

        test_loss = []
        epoch_loss = 0
        test_acc = []
        total_loss, total_accuracy, total_samples = 0, 0, 0
        corr, total = 0, 0


        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            # calculate loss
            loss = criterion(outputs, labels)
            loss = reduce_value(loss, average=True)  #  在单GPU中不起作用，多GPU时，获得所有GPU的loss的均值。
            test_loss.append(loss.item())   # 平均了所有进程的loss, save average loss for each batch
            
            # calculate accuracy
            #_, predicted = torch.max(outputs, 1)
            ## torch.max(outputs, 1) will return (max_value, max_indice) in outputs at dim-1 axis. 
            #corr += (predicted == labels).sum().item()
            #total += labels.size(0)
            #acc0 = corr / total
            #correct = torch.tensor(acc0, dtype=torch.float32).to('cuda').to_dense()
            #correct = reduce_value(correct, average=True)

            #test_acc.append(correct.item())
            #epoch_acc = (np.mean(test_acc)).tolist()

                
    epoch_loss = (np.mean(test_loss)).tolist()
    #epoch_acc = (np.mean(test_acc)).tolist()
    #print(f'Test Loss:{epoch_loss}, accuracy : {epoch_acc}')
    print(f'Test Loss:{epoch_loss}.')

    return epoch_loss
            

