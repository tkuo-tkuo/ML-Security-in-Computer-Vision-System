import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from binary_MNIST_models import NaiveC, NormalC, CNN
from MNIST_models import NaiveC, NormalC, CNN
from utils import *

class PropertyInferenceInterface():

    def __init__(self):
        MNIST_train_dataset = torchvision.datasets.MNIST(root='./data', transform=transforms.ToTensor(), train=True, download=True)
        MNIST_test_dataset = torchvision.datasets.MNIST(root='./data', transform=transforms.ToTensor(), train=False, download=True)
        self.train_loader = torch.utils.data.DataLoader(dataset=MNIST_train_dataset, batch_size=50000, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(dataset=MNIST_test_dataset, batch_size=10000, shuffle=False)
        self.meta_params = {}
        self.train_dataset = None
        self.test_dataset = None
        self.model = None

        self.LPs_set = None        

    # Debug Information
    def print_meta_params(self):
        for key in self.meta_params.keys():
            print((str(key)).ljust(20), ':', self.meta_params[key])

    # Debug Information
    def print_dataset_shape(self):
        print('Train dataset')
        print(self.train_dataset[0].shape, self.train_dataset[1].shape)
        print('Test dataset')
        print(self.test_dataset[0].shape, self.test_dataset[1].shape)
        
    # Debug Information
    def print_some_samples(self):
        import math
        import matplotlib.pyplot as plt
        X, _ = self.train_dataset
        for i in range(len(X)):
            if i < 32:
                data = (X[i]).reshape(28, 28)
                plt.subplot(8, 4, (1+i))
                plt.axis('off')
                plt.imshow(data, cmap='gray')

        plt.show()

    def set_meta_params(self, meta_params):
        self.meta_params = meta_params

    def prepare_dataset(self):
        self._create_train_dataset()
        self._create_test_dataset()

    def _create_train_dataset(self):
        X, Y = None, None 
        
        for (samples, labels) in self.train_loader:
            if self.meta_params['flatten'] is True:
                X = samples.reshape(-1, 784).numpy()
            else:
                X = samples.numpy()
            Y = (labels.numpy())
        
            # Prepare subset 
            if 'size_of_train_set' in list(self.meta_params.keys()):
                (X, Y) = extract_subset(X, Y, self.meta_params['size_of_train_set'])
            break
        
        self.train_dataset = (X, Y)

    def _create_test_dataset(self):
        X, Y = None, None 

        for (samples, labels) in self.test_loader:
            if self.meta_params['flatten'] is True:
                X = samples.reshape(-1, 784).numpy()
            else:
                X = samples.numpy()
            Y = (labels.numpy())
        
            # Prepare subset 
            if 'size_of_test_set' in list(self.meta_params.keys()):
                (X, Y) = extract_subset(X, Y, self.meta_params['size_of_test_set'])
            break
        
        self.test_dataset = (X, Y)

    def set_model(self, model):
        self.model = model

    def generate_model(self, num_of_epochs=15):
        if self.meta_params['model_type'] == 'naive':
            model = NaiveC()
        elif self.meta_params['model_type'] == 'normal':
            model = NormalC()
        elif self.meta_params['model_type'] == 'CNN':
            model = CNN()

        X, Y = self.train_dataset
        
        # Training
        loss_func, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(num_of_epochs):
            print(epoch)
            for idx, data in enumerate(X):

                # Transform from numpy to torch & correct the shape (expand dimension) and type (float32 and int64)
                data = torch.from_numpy(np.expand_dims(data, axis=0).astype(np.float32))
                label = torch.from_numpy(np.array([Y[idx]]).astype(np.int64))
        
                # Forwarding
                prediction = model.forward(data)
                loss = loss_func(prediction, label)

                # Optimization (back-propogation)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        self.model = model

    def eval_model(self, dataset_type):
        if dataset_type == 'train':
            X, Y = self.train_dataset
        else: 
            X, Y = self.test_dataset

        model = self.model
        datas = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(Y.astype(np.int64))
        
        # Forwarding
        outputs = model.forward(datas).detach().numpy()
        predictions = np.argmax(outputs, axis=1)

        total = labels.shape[0]
        correct = (predictions == labels.numpy()).sum().item()
        acc = correct/total
            
        print('Model (', dataset_type, ') accurancy:', acc)
        return acc

    def generate_LPs(self):
        X, Y = self.train_dataset

        LPs_set = []
        for i in range(10):
            LPs_set.append([])
        for i in range(10):
            for _ in range(4):
                LPs_set[i].append([])
            
        for i in range(len(X)):
            x, y = X[i], Y[i]
            LPs = extract_all_LP(self.model, self.meta_params['model_type'], x)
            for i in range(len(LPs)):
                (LPs_set[y])[i].append(LPs[i])

        self.LPs_set = LPs_set

    def print_LPs(self):
        LPs_set = np.array(self.LPs_set)
        for i in range(10):
            print('LPs shape of', (i+1), 'th class:', LPs_set[i].shape)
            for LPs in LPs_set[i]:
                print((np.array(LPs)).shape)

    def property_match(self, x, y, verbose=True, alpha=None,):
        Py = self.LPs_set[y]
        LPs = extract_all_LP(self.model, self.meta_params['model_type'], x)
        # prob_LP1 = np.sum(LP1s, axis=0) / (LP1s.shape[0])

        #############################################
        # Original Method 
        #############################################
        LP_status = []
        for i in range(len(LPs)):
            LP_i = Py[i]
            p_i = LPs[i]
            
            status = 'adversarial'
            if p_i in LP_i:
                status = 'benign'
            LP_status.append(status)

        result = 0
        # if not ('adversarial' in LP_status):
        if 'benign' == LP_status[3]:
            result = 1
            
        if verbose:
            if result == 1:
                print(LP_status, 'benign')
            else:
                print(LP_status, 'adversarial')
        #############################################

       #############################################
        # Experimental: Method 1 & 2
        #############################################
        # diff = Prob_Py - precondition
        # abs_diff = np.absolute(diff)
        # abs_diff[abs_diff < 0.1] = 0
        # risk_score = np.sum(abs_diff) 
        # print('score:', risk_score)

        # result = 0
        # if risk_score < 10:
        #     result = 1
        #############################################


        #############################################
        # Experimental: Method 3 & 4
        #############################################
        # offset = 0.45
        # weights = np.array(Prob_Py) 
        # weights[weights <= (0.5-offset)] = -1
        # weights[weights >= (0.5+offset)] = -1
        # weights[weights != -1] = 0
        # weights[weights == -1] = 1

        # Prob_Py[Prob_Py==1.0] = 1.0 - 1/(Prob_Py.shape[0] + 1)
        # Prob_Py[Prob_Py==0.0] = 0.0 + 1/(Prob_Py.shape[0] + 1)
        # Prob_Py_ = 1 - Prob_Py
        # np_precondition = np.array(precondition)

        # benign_prob = 1
        # for i, ele in enumerate(np_precondition):
        #     # if weights[i] == 0:
        #     #     continue

        #     if ele == 1:
        #         benign_prob *= Prob_Py[i]
        #     else:
        #         benign_prob *= Prob_Py_[i]

        # # print('benign_prob:', benign_prob)

        # if alpha == None:
        #     alpha = 5e-8

        # # 0 indicates adversarial 
        # # 1 indicates benign 
        # result = 0
        # if benign_prob > alpha:
        #     result = 1

        # if verbose: 
        #     print(benign_prob, alpha, result)
        #############################################

        return (result, LP_status)

    def evaluate_algorithm_on_test_set(self, alpha=None, verbose=True):
        # self._double_check_on_train_set()
        benign_detect_ratio, benign_LPs = self._evaluate_benign_samples(alpha, verbose)
        adversarial_detect_ratio, adversarial_LPs = self._evaluate_adversarial_samples(alpha, verbose)
        return (benign_detect_ratio, adversarial_detect_ratio), (benign_LPs, adversarial_LPs)

    def _evaluate_benign_samples(self, alpha, verbose):
        LPs = []

        test_X, test_Y = self.test_dataset
        num_of_count, valid_count = len(test_X), 0
        for i in range(num_of_count):
            print('benign', i)
            x, y = test_X[i], test_Y[i]
            
            # Use y_
            output = self.model.forward(torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32)))
            y_ = (output.max(1, keepdim=True)[1]).item() 
            if y_ != y:
                num_of_count -= 1
                continue

            if verbose:
                print('Benign input matching...')

            result, LP_status = self.property_match(x, y_, verbose, alpha) # y'
            LPs.append(LP_status)
            valid_count += result

        if verbose:
            print('Evaluate on benign samples with test set')
            print(valid_count, num_of_count, (valid_count/num_of_count))

        return (valid_count/num_of_count), LPs

    def _evaluate_adversarial_samples(self, alpha, verbose):
        LPs = []

        import attacker
        if self.meta_params['adv_attack'] == 'i_FGSM':
            A = attacker.iterative_FGSM_attacker()
        elif self.meta_params['adv_attack'] == 'JSMA':
            A = attacker.JSMA_attacker()
        elif self.meta_params['adv_attack'] == 'CW_L2':
            A = attacker.CW_L2_attacker()
        else:
            A = NotImplemented

        test_X, test_Y = self.test_dataset
        model = self.model

        num_of_count = len(test_X)
        valid_count = 0        
        success_count = 0
        for i in range(num_of_count):
            print('Conduct', i, 'th attack:', self.meta_params['adv_attack'])
            x, y = test_X[i], test_Y[i]

            # Use y_
            output = self.model.forward(torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32)))
            y_ = (output.max(1, keepdim=True)[1]).item() 
            if y_ != y:
                num_of_count -= 1
                continue

            is_attack_successful = False 
            epsilon = 0 
            while (not is_attack_successful):
                epsilon += 0.01
                adv_x, success_indicator = A.create_adv_input(x, y, model, epsilon)
                if success_indicator == 1:
                    success_count += success_indicator
                    is_attack_successful = True
                    adv_x = adv_x.detach().numpy()
                    adv_x = adv_x[0]

                    # if i == 0 and verbose:
                    #     import matplotlib.pyplot as plt
                    #     img = adv_x.reshape(28, 28)
                    #     plt.imshow(img, cmap='gray')
                    #     plt.show()

                    #     img = x.reshape(28, 28)
                    #     plt.imshow(img, cmap='gray')
                    #     plt.show()

            output = self.model.forward(torch.from_numpy(np.expand_dims(adv_x, axis=0).astype(np.float32)))
            y_ = (output.max(1, keepdim=True)[1]).item()
            if verbose:
                print('Adversarial input matching...')

            result, LP_status = self.property_match(adv_x, y_, verbose, alpha) # y'
            LPs.append(LP_status)
            valid_count += result
        
        assert success_count == num_of_count 
        if verbose:
            print('Evaluate on adversarial samples with test set')
            print((num_of_count - valid_count), num_of_count, (num_of_count - valid_count)/num_of_count)
            
        return ((num_of_count - valid_count)/num_of_count), LPs
