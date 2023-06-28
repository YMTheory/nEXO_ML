import numpy as np
import time
import torch
import torch.nn as nn
import os

"""
Manipulating network in this module: training, test...
"""
best_acc = 0

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = lr/np.exp(epoch/10)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
 


def train(net, criterion, optimizer, trainloader, device, epoch, grad_clip=False):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_value_(net.parameters(), grad_clip)

        optimizer.step()

        loss = reduce_value(loss, average=True) # not work in one GPU, get average loss in multi GPU.
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, '/', len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/len(trainloader), 100.*correct/total


def test(net, criterion, testloader, device, epoch, modelpath, modelfile, saveall=False):
    
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    score = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss = reduce_value(loss, average=True)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            softmax = nn.Softmax(dim=0)
            for m in range(outputs.size(0)):
                score.append([softmax(outputs[m])[1].item(), targets[m].item()])
                # score.append([outputs[m][1].item(), targets[m].item()])
            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    is_better = False
    acc = 100.*correct/total
    
    # If we want to save all training records
    if saveall:
        print ('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(modelpath):
            os.mkdir(modelpath)
        torch.save(state, modelpath + modelfile[:-3]+"_"+str(epoch)+modelfile[-3:])
        torch.save(state, modelpath + modelfile )
        best_acc = acc
    # Otherwise only save the best one
    elif acc > best_acc:
        print ('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(modelpath):
            os.mkdir(modelpath)
        torch.save(state, modelpath + modelfile[:-3]+"_"+str(epoch)+modelfile[-3:])
        torch.save(state, modelpath + modelfile )
        best_acc = acc
        
    return test_loss/len(testloader), 100.*correct/total, score




def train_net(start_epoch, epochs, device, lr, net, criterion, optimizer, train_loader, validation_loader, resume, saveall, loss_acc_path, loss_acc_file, modelpath, modelfile):

    # Numpy arrays for loss and accuracy, if resume from checkpoint then read the previous results.
    if resume and os.path.exists(loss_acc_file):
        arrays_resumed = np.load(loss_acc_file, allow_pickle=True)
        y_train_loss = arrays_resumed[0]
        y_train_acc  = arrays_resumed[1]
        y_valid_loss = arrays_resumed[2]
        y_valid_acc  = arrays_resumed[3]
        test_score   = arrays_resumed[4].tolist()
    else:
        y_train_loss = np.array([])
        y_train_acc  = np.array([])
        y_valid_loss = np.array([])
        y_valid_acc  = np.array([])
        test_score   = []

    # Start to train the network:
    for epoch in range(start_epoch, start_epoch + epochs):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, lr)
        iterout = "\nEpoch [%d]: "%(epoch)

        for param_group in optimizer.param_groups:
            iterout += "lr=%.3e"%(param_group['lr'])
            print(iterout)

            try:
                train_ave_loss, train_ave_acc = train(net, criterion, optimizer, train_loader, device, epoch)
            except Exception as e:
                print("Error in training routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
            print("Train[%d]: Result* Loss %.3f\t Accuracy: %.3f"%(epoch, train_ave_loss, train_ave_acc))
            y_train_loss = np.append(y_train_loss, train_ave_loss)
            y_train_acc = np.append(y_train_acc, train_ave_acc)


            # Evaluate on validationset
            try:
                valid_loss, prec1, score = test(net, criterion, validation_loader, device, epoch, modelpath, modelfile, saveall=saveall)
            except Exception as e:
                print("Error in validation routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
                
            print("Test[%d]: Result* Loss %.3f\t Precision: %.3f"%(epoch, valid_loss, prec1))
            
            test_score.append(score)
            y_valid_loss = np.append(y_valid_loss, valid_loss)
            y_valid_acc = np.append(y_valid_acc, prec1)
            
            np.save(loss_acc_path+loss_acc_file, np.array([y_train_loss, y_train_acc, y_valid_loss, y_valid_acc, test_score], dtype=object))
        stop_time = time.time()
        print(f"TIME collapsed in epoch {epoch} is {stop_time - start_time}Msec.")



def resume_model(net, device, modelpath, modelfile):
    # Load checkpoint
    print("===> Resuming from checkpoint...")
    assert os.path.isdir(modelpath), "Error: no checkpoint directory found!"
    if device == 'cuda':
        checkpoint = torch.load(modelpath + modelfile)
    else:
        checkpoint = torch.load(modelpath + modelfile, map_location=torch.device('cpu') )
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

    return net, best_acc, start_epoch
















