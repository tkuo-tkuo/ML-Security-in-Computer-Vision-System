import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import copy
import random

import time

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(3*3*32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # (24, 24, 16)
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # (20, 20, 16) -> (10, 10, 16)
        x = F.relu(F.max_pool2d(self.conv3(x), 2)) # (6, 6, 32) -> (3, 3, 32)
        x = x.view(-1, 3*3*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Guard(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        
        self.kernel_size = 1
        
        self.pre_l1_conv1 = nn.Conv2d(16, 8, 1)
        self.pre_l2_conv1 = nn.Conv2d(16, 8, 1)
        self.pre_l3_conv1 = nn.Conv2d(32, 8, 1)
        
        self.xh1 = nn.Conv2d(8, 8, 1)
        self.xh2 = nn.Conv2d(8, 8, 1)
        self.xh3 = nn.Conv2d(8, 8, 1)
        self.xh4 = nn.Linear(64, 64)
        
        self.hh12 = nn.Conv2d(8, 8, 15)
        self.hh23 = nn.Conv2d(8, 8, 8)
        self.hh34 = nn.Linear(8*3*3, 64)
                
        self.fc1 = nn.Linear(5544, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, f1, f2, f3, f4):
        x1 = self.relu(self.pre_l1_conv1(f1))   # 1, 8, 24, 24 
        x2 = self.relu(self.pre_l2_conv1(f2))   # 1, 8, 10, 10
        x3 = self.relu(self.pre_l3_conv1(f3))   # 1, 8, 3, 3
        x4 = f4.view(-1, 64)                    # 1, 64

        LN = nn.GroupNorm(8, 8)
        x1 = LN(x1)
        LN = nn.GroupNorm(8, 8)
        x2 = LN(x2)
        LN = nn.GroupNorm(8, 8)
        x3 = LN(x3)
        LN = nn.LayerNorm(x4.size()[1:])
        x4 = LN(x4)
        
        h1 = self.relu(self.xh1(x1)) 
        h1 = F.dropout2d(h1, p=0.2)
        
        h2 = self.relu(torch.add(self.hh12(h1), self.xh2(x2)))
        h2 = F.dropout2d(h2, p=0.2)

        h3 = self.relu(torch.add(self.hh23(h2), self.xh3(x3)))
        h3 = F.dropout2d(h3, p=0.2)

        h3 = h3.view(-1, 8*3*3)
        h4 = self.relu(torch.add(self.hh34(h3), self.xh4(x4)))
        h4 = F.dropout(h4, p=0.2)

        fc0 = torch.cat([h1.view(-1, 8*24*24), h2.view(-1, 8*10*10), h3.view(-1, 8*3*3), h4], 1)
        fc1 = self.relu(self.fc1(fc0))
        fc1 = F.dropout(fc1, p=0.2)
        outputs = self.fc2(fc1)

        return outputs

class AssemebleGuard(nn.Module):
    def __init__(self, sub_guards):
        super().__init__()
        self.sub_guard_0 = sub_guards[0]
        self.sub_guard_1 = sub_guards[1]
        self.sub_guard_2 = sub_guards[2]
        self.sub_guard_3 = sub_guards[3]
        self.sub_guard_4 = sub_guards[4]
        self.sub_guard_5 = sub_guards[5]
        self.sub_guard_6 = sub_guards[6]
        self.sub_guard_7 = sub_guards[7]
        self.sub_guard_8 = sub_guards[8]
        self.sub_guard_9 = sub_guards[9]
        self.to_final_state = nn.Linear(20, 2)

    def forward(self, f1, f2, f3, f4):
        outputs0 = self.sub_guard_0(f1, f2, f3, f4)
        outputs1 = self.sub_guard_1(f1, f2, f3, f4)
        outputs2 = self.sub_guard_2(f1, f2, f3, f4)
        outputs3 = self.sub_guard_3(f1, f2, f3, f4)
        outputs4 = self.sub_guard_4(f1, f2, f3, f4)
        outputs5 = self.sub_guard_5(f1, f2, f3, f4)
        outputs6 = self.sub_guard_6(f1, f2, f3, f4)
        outputs7 = self.sub_guard_7(f1, f2, f3, f4)
        outputs8 = self.sub_guard_8(f1, f2, f3, f4)
        outputs9 = self.sub_guard_9(f1, f2, f3, f4)
        combined_outputs = torch.cat((outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8, outputs9), 1)
        outputs = self.to_final_state(combined_outputs)

        return outputs

class LSTM(nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim, hidden_layer_num, batch_size, classes_num):
        super().__init__()
        self.input_feature_dim = input_feature_dim
        self.hidden_feature_dim = hidden_feature_dim
        self.hidden_layer_num = hidden_layer_num
        self.batch_size = batch_size
        self.classes_num = classes_num
        
        self.lstm = nn.LSTM(self.input_feature_dim, self.hidden_feature_dim, self.hidden_layer_num)
        self.fc1 = nn.Linear(self.hidden_feature_dim, 8)
        self.fc2 = nn.Linear(8, self.classes_num)
        self.lrelu = nn.LeakyReLU()

    def init_hidden(self):
        h0 = torch.randn(self.hidden_layer_num,self.batch_size,self.hidden_feature_dim)
        c0 = torch.randn(self.hidden_layer_num,self.batch_size,self.hidden_feature_dim)
        return (h0, c0)
        
    def forward(self, inputs):
        hidden = self.init_hidden()
        output, _ = self.lstm(inputs, hidden)
        f1 = self.lrelu(self.fc1(output[-1]))
        f2 = self.fc2(f1)
        return f2

def store_model(model, model_name):
    torch.save(model, model_name)

def load_model(model_name):
    return torch.load(model_name)

def train(train_dataset, model, loss_func, opt, num_of_epochs):
    model.train()
    X, Y = train_dataset

    for epoch in range(num_of_epochs):
        print('Generate CNN model, epoch:', epoch+1, '...')
        for idx, data in enumerate(X):
            # Transform from numpy to torch & correct the shape (expand dimension) and type (float32 and int64)
            data = torch.from_numpy(np.expand_dims(data, axis=0).astype(np.float32))
            label = torch.from_numpy(np.array([Y[idx]]).astype(np.int64))

            # Forwarding
            outputs = model.forward(data)
            loss = loss_func(outputs, label)

            # Optimization (back-propogation)
            if random.random() < 0.6:
                opt.zero_grad()
                loss.backward()
                opt.step()

        print('acc:', eval_model(model, train_dataset))

    return model

def eval_model(model, dataset):
    X, Y = dataset
    correct_count = 0
    for idx, data in enumerate(X):
        # Transform from numpy to torch & correct the shape (expand dimension) and type (float32 and int64)
        data = torch.from_numpy(np.expand_dims(data, axis=0).astype(np.float32))
        label = torch.from_numpy(np.array([Y[idx]]).astype(np.int64))

        # Forwarding
        outputs = model.forward(data).detach().numpy()
        prediction = np.argmax(outputs, axis=1)

        is_correct = (prediction == label.numpy()).item()
        if is_correct: correct_count += 1

    acc = correct_count/len(X)
    return acc

def preprocess(x):
    f1 = torch.from_numpy(np.array(x[0])).float()
    f2 = torch.from_numpy(np.array(x[1])).float()
    f3 = torch.from_numpy(np.array(x[2])).float()
    f4 = np.expand_dims(x[3], axis=2)
    f4 = torch.from_numpy(f4).float()
    return f1, f2, f3, f4
        
def train_guard_model(guard_model, set_of_train_dataset, set_of_test_dataset, adv_types, epoches):
    loss_func, optimizer = nn.BCEWithLogitsLoss(), torch.optim.Adam(guard_model.parameters(), lr=1e-3)
    train_accs, test_accs, losses = [], [], []
    set_train_sub_accs, set_test_sub_accs = [], []

    for epoch in range(epoches):
        start = time.clock()

        total_loss = None 
        # labeling ...
        train_dataset, train_labels = [], []
        for dataset, adv_type in zip(set_of_train_dataset, adv_types):
            for singatures in dataset:
                if adv_type == 'None': 
                    for _ in range(2):
                        train_dataset.append(singatures)
                        label = torch.from_numpy(np.array([[1, 0]])).float()
                        train_labels.append(label)

                else: 
                    train_dataset.append(singatures)
                    label = torch.from_numpy(np.array([[0, 1]])).float()
                    train_labels.append(label)
  
        # shuffling 
        shuffle_indexs = np.arange(len(train_dataset))
        np.random.shuffle(shuffle_indexs)

        # training 
        for index in shuffle_indexs:
            singatures, label = train_dataset[index], train_labels[index]
            f1, f2, f3, f4 = preprocess(singatures)
            outputs = guard_model.forward(f1, f2, f3, f4)

            # for recording the training process 
            loss = loss_func(outputs, label)
            if total_loss is None: total_loss = loss 
            else: total_loss += loss
            
            # Optimization (back-propogation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_acc, train_sub_accs = test_guard_model(guard_model, set_of_train_dataset, adv_types, verbose=True)
        print()
        test_acc, test_sub_accs = test_guard_model(guard_model, set_of_test_dataset, adv_types, verbose=True)
        print()
        print('epoch:', (epoch+1), 'loss:', total_loss.item())    
        print('acc (train):', train_acc)
        print('acc (test):', test_acc)
        print()
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        set_train_sub_accs.append(train_sub_accs)
        set_test_sub_accs.append(test_sub_accs)
        losses.append(total_loss)

        end = time.clock()
        print('one epoch training time in seconds', end-start)

    return train_accs, test_accs, losses, set_train_sub_accs, set_test_sub_accs

def test_guard_model(guard_model, set_of_test_dataset, adv_types, verbose=True):
    total_train_correct_count, total_train_count = 0, 0 
    sub_accs = []
    for test_dataset, adv_type in zip(set_of_test_dataset, adv_types):
        current_count = 0
        for singatures in test_dataset:
            f1, f2, f3, f4 = preprocess(singatures)
            outputs = guard_model.forward(f1, f2, f3, f4)
            if adv_type == 'None': label = torch.from_numpy(np.array([[1, 0]])).float()
            else: label = torch.from_numpy(np.array([[0, 1]])).float()

            prediction = (outputs.max(1, keepdim=True)[1]).item()     
            if adv_type == 'None': 
                if (prediction == 0): 
                    current_count += 1
            else: 
                if (prediction == 1): 
                    current_count += 1
            
        # record the current train set acc
        if verbose: 
            if adv_type == 'None': 
                print('benign correct:', current_count, '/', len(test_dataset))
            else:
                print('adv (', adv_type, ') correct:', current_count, '/', len(test_dataset))

        sub_accs.append(current_count/len(test_dataset))
        total_train_correct_count += current_count
        total_train_count += len(test_dataset)

    acc = total_train_correct_count/total_train_count
    if verbose: 
        print('acc:', acc)

    return acc, sub_accs

def LSTM_preprocess(x):
    f1 = torch.from_numpy(np.array(x[0])).float()
    f1 = f1.view(-1)

    f2 = torch.from_numpy(np.array(x[1])).float()
    f2 = f2.view(-1)
    padding = torch.zeros(f1.shape[0]-f2.shape[0])
    f2 = torch.cat([f2, padding])
    
    f3 = torch.from_numpy(np.array(x[2])).float()
    f3 = f3.view(-1)
    padding = torch.zeros(f1.shape[0]-f3.shape[0])
    f3 = torch.cat([f3, padding])
    
    f4 = np.expand_dims(x[3], axis=2)
    f4 = torch.from_numpy(f4).float()
    f4 = f4.view(-1)
    padding = torch.zeros(f1.shape[0]-f4.shape[0])
    f4 = torch.cat([f4, padding])

    # inputs = torch.cat([f1, f2, f3, f4]).reshape(4, 1, -1)
    inputs = torch.cat([f4, f3, f2, f1]).reshape(4, 1, -1)
    return inputs

def train_LSTM_PPRD(lstm, set_of_train_dataset, set_of_test_dataset, adv_types, epoches):
    '''
    Pytorch’s LSTM expects all of its inputs to be 3D tensors.

    The first axis is the sequence itself, 
    the second indexes instances in the mini-batch, 
    and the third indexes elements of the input.
    '''
    optimizer, loss_func = torch.optim.Adam(lstm.parameters()), nn.CrossEntropyLoss()
    train_accs, test_accs, losses = [], [], []
    set_train_sub_accs, set_test_sub_accs = [], []

    # labeling 
    train_dataset, train_labels = [], []
    for dataset, adv_type in zip(set_of_train_dataset, adv_types):
        for singatures in dataset:
            if adv_type == 'None': 
                train_dataset.append(singatures)
                label = torch.from_numpy(np.array([[1, 0]])).float()
                train_labels.append(label)

            else: 
                train_dataset.append(singatures)
                label = torch.from_numpy(np.array([[0, 1]])).float()
                train_labels.append(label)
                
    test_dataset, test_labels = [], []
    for dataset, adv_type in zip(set_of_test_dataset, adv_types):
        for singatures in dataset:
            if adv_type == 'None': 
                test_dataset.append(singatures)
                label = torch.from_numpy(np.array([[1, 0]])).float()
                test_labels.append(label)

            else: 
                test_dataset.append(singatures)
                label = torch.from_numpy(np.array([[0, 1]])).float()
                test_labels.append(label)

    for epoch in range(epoches):
        # shuffling 
        shuffle_indexs = np.arange(len(train_dataset))
        np.random.shuffle(shuffle_indexs)
        
        total_loss = 0 
        for index in shuffle_indexs:
            data, label = train_dataset[index], train_labels[index]
            inputs = LSTM_preprocess(data)
            label = np.argmax(label, axis=1)

            output = lstm(inputs)
            loss = loss_func(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if total_loss is None: total_loss = loss
            else: total_loss += loss
            print(loss)

                
        print('epoch:', epoch+1, 'loss:', total_loss.item())
        train_acc, train_sub_accs = test_LSTM_PPRD(lstm, set_of_train_dataset, adv_types, verbose=False)
        test_acc, test_sub_accs = test_LSTM_PPRD(lstm, set_of_test_dataset, adv_types, verbose=False)
        print('acc (train):', train_acc)
        print('acc (test):', test_acc)

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        set_train_sub_accs.append(train_sub_accs)
        set_test_sub_accs.append(test_sub_accs)
        losses.append(total_loss)
        
        model_name = 'LSTM_PPRD_'+str(epoch+1)
        torch.save(lstm, model_name)

    return train_accs, test_accs, losses, set_train_sub_accs, set_test_sub_accs
    

def test_LSTM_PPRD(lstm, set_of_test_dataset, adv_types, verbose=True):
    total_train_correct_count, total_train_count = 0, 0 
    sub_accs = []
    for test_dataset, adv_type in zip(set_of_test_dataset, adv_types):
        current_count = 0
        for singatures in test_dataset:
            inputs = LSTM_preprocess(singatures)
            outputs = lstm.forward(inputs)
            if adv_type == 'None': label = torch.from_numpy(np.array([[1, 0]])).float()
            else: label = torch.from_numpy(np.array([[0, 1]])).float()
            label = np.argmax(label, axis=1)

            prediction = (outputs.max(1, keepdim=True)[1]).item()     
            if adv_type == 'None': 
                if (prediction == 0): 
                    current_count += 1
            else: 
                if (prediction == 1): 
                    current_count += 1
            
        # record the current train set acc
        if verbose: 
            if adv_type == 'None': 
                print('benign correct:', current_count, '/', len(test_dataset))
            else:
                print('adv (', adv_type, ') correct:', current_count, '/', len(test_dataset))

        sub_accs.append(current_count/len(test_dataset))
        total_train_correct_count += current_count
        total_train_count += len(test_dataset)

    acc = total_train_correct_count/total_train_count
    if verbose: 
        print('acc:', acc)

    return acc, sub_accs