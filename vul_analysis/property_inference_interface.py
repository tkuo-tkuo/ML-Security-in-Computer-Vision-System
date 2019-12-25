import math
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from binary_MNIST_models import NaiveC, NormalC, CNN
from MNIST_models import NaiveC, NormalC, CNN, robustified_FC, robustified_CNN
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
        self.robustified_model = None 

        self.LPs_set = None        
        self.differentation_lines = [1, 1, 1, 1] 

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

    def store_model(self, model_name):
        torch.save(self.model, model_name)

    def load_model(self, model_name):
        self.model = torch.load(model_name)

    def generate_twisted_model(self, model_type, num_of_epochs=10, dropout_rate=None):
        ''' 
        Currently, we only have CNN => add FC in the future 
        '''

        # Create an untrained robustified CNN
        self.robustified_model = robustified_CNN(dropout_rate)
        self.model.train()
        self.robustified_model.train()

        # Load the original model 
        import copy
        original_state = copy.deepcopy(self.model.state_dict())
        for name, param in self.robustified_model.state_dict().items():
            if name not in original_state:
                print('Not in original_state:', name)
                continue
            
            param.copy_(original_state[name])

        # Retraining 
        X, Y = self.train_dataset
        model = self.robustified_model
        model.train()

        loss_func, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(num_of_epochs):
            # print(epoch)
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

    def generate_robustified_model(self, model_type, num_of_epochs=15, dropout_rate=None):
        X, Y = self.train_dataset

        if model_type == 'FC':
            model = robustified_FC(dropout_rate)
        elif model_type == 'CNN':
            model = robustified_CNN(dropout_rate)
        else: 
            pass 

        model.train()
        # Training
        loss_func, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(num_of_epochs):
            # print(epoch)
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
        
        self.robustified_model = model

    def generate_model(self, num_of_epochs=15):
        if self.meta_params['model_type'] == 'naive':
            model = NaiveC()
        elif self.meta_params['model_type'] == 'normal':
            model = NormalC()
        elif self.meta_params['model_type'] == 'CNN':
            model = CNN()

        X, Y = self.train_dataset
        
        model.train()
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

    def eval_model(self, dataset_type, on_robustified_model=False):
        if dataset_type == 'train':
            X, Y = self.train_dataset
        else: 
            X, Y = self.test_dataset

        if on_robustified_model:
            model = self.robustified_model
        else:
            model = self.model

        model.eval()
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
            for _ in range(self.meta_params['num_of_LPs']):
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

    def property_match(self, x, y, verbose=True):
        Py = self.LPs_set[y]
        LPs = extract_all_LP(self.model, self.meta_params['model_type'], x)

        #############################################
        # Original Method 
        # result == 1 -> the given input is considered as 'benign'
        # result == 0 -> the given input is considered as 'adversarial'
        #############################################
        '''
        LP_status = []
        LP_risk_score = []

        for i in range(len(LPs)):
            LP_i = Py[i]
            p_i = LPs[i]
            
            status = 'adversarial'
            if p_i in LP_i:
                status = 'benign'

            LP_status.append(status)
            if status == 'benign':
                LP_risk_score.append(1)
            else:
                LP_risk_score.append(0)

        
        result = 1
        if 'adversarial' == LP_status[0] and 'adversarial' == LP_status[1]:
            result = 0

        if verbose:
            if result == 1:
                print(LP_status, 'benign')
            else:
                print(LP_status, 'adversarial')

        return (result, LP_status, LP_risk_score)
        '''
        #############################################

        #############################################
        # Experimental: Method 1 & 2
        #############################################
        LP_status = []
        LP_risk_score = []
        differentiation_lines = self.differentation_lines
        for i in range(len(LPs)):
            differentiation_line = differentiation_lines[i]
            LP_i = np.array(Py[i])
            p_i = np.array(LPs[i])

            prob_LP_i = np.sum(LP_i, axis=0) / LP_i.shape[0]
            diff = prob_LP_i - p_i
            abs_diff = np.absolute(diff)
            # abs_diff[abs_diff<0.9] = 0
            risk_score = np.sum(abs_diff)
            
            status = 'adversarial'
            if risk_score < differentiation_line:
                status = 'benign'

            LP_status.append(status)
            LP_risk_score.append(risk_score)

        result = 0
        if 'benign' == LP_status[0] and 'benign' == LP_status[1] and 'benign' == LP_status[2]:
            result = 1    

        '''
        if verbose:
            if result == 1:
                print(LP_status, 'benign')
            else:
                print(LP_status, 'adversarial')
        '''

        return (result, LP_status, LP_risk_score)
        #############################################


        #############################################
        # Experimental: Method 3 & 4
        #############################################
        '''
        LP_status = []
        LP_risk_score = []
        prob_diff_lines = [-800, -600, -150, 1e-4]
        for i in range(len(LPs)):
            prob_diff_line = prob_diff_lines[i]
            LP_i = np.array(Py[i])
            p_i = np.array(LPs[i])

            # compute probability of LP_i
            prob_LP_i = np.sum(LP_i, axis=0) / LP_i.shape[0]

            # To avoid 0 probability in either prob_LP_i or prob_LP_i_0
            prob_LP_i[prob_LP_i==1.0] = 1.0 - (1/(prob_LP_i.shape[0] + 1))
            prob_LP_i[prob_LP_i==0.0] = 0.0 + (1/(prob_LP_i.shape[0] + 1))
            prob_LP_i_0 = 1 - prob_LP_i

            # This section is for Method 4
            # offset = 0.1
            # weights = np.array(prob_LP_i) 
            # weights[weights <= (0.5-offset)] = -1
            # weights[weights >= (0.5+offset)] = -1
            # weights[weights != -1] = 0
            # weights[weights == -1] = 1
            # 

            B_prob = 0
            for i, neuron_activation in enumerate(p_i):
                # This section is for Method 4
                # use weights[i] for Method 4
                #
                if neuron_activation == 1:
                    B_prob += math.log(prob_LP_i[i]) * weights[i]
                else:
                    B_prob += math.log(prob_LP_i_0[i]) * weights[i]

            status = 'adversarial'
            if B_prob > prob_diff_line:
                status = 'benign'

            LP_status.append(status)
            LP_risk_score.append(B_prob)

        result = 0
        if 'benign' == LP_status[0] and 'benign' == LP_status[1] and 'benign' == LP_status[2]:
            result = 1    

        if verbose:
            if result == 1:
                print(LP_status, 'benign')
            else:
                print(LP_status, 'adversarial')

        return (result, LP_status, LP_risk_score)
        '''
        #############################################

    def evaluate_algorithm_on_test_set(self, verbose=True):
        self._set_differentation_lines(95)
        B_num_count, B_correct_count, B_valid_count, B_LPs, B_LPs_score = self._evaluate_benign_samples(verbose)
        A_num_count, A_correct_count, A_valid_count, A_LPs, A_LPs_score = self._evaluate_adversarial_samples(verbose)
        B_detect_ratio, A_detect_ratio = B_valid_count/B_correct_count, A_valid_count/A_correct_count
        return (B_detect_ratio, A_detect_ratio), (B_LPs, A_LPs), (B_LPs_score, A_LPs_score)

    def _set_differentation_lines(self, qr):
        ''' Private function
        - Set the differentiation lines according to training dataset,
        - Apply differentation lines on B (normal test samples) and A (adversarial test samples)
        '''

        # Load (train) dataset and model 
        X, Y = self.train_dataset 
        model = (self.model).eval()

        # Create intermediate variables 
        LPs_score = []

        for i in range(len(X)):
            x, y = X[i], Y[i]
            
            # Filter out samples can not be correctly classified by the given model 
            output = model.forward(torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32)))
            y_ = (output.max(1, keepdim=True)[1]).item() 
            if y_ != y:
                continue

            # Collect LP_risk_score among train dataset 
            _, _, LP_risk_score = self.property_match(x, y_, False) 
            LPs_score.append(LP_risk_score)

        # Compute differentation lines 
        LPs_score = np.array(LPs_score)
        differentation_lines = []
        for i in range(LPs_score.shape[1]):
            LP_score = LPs_score[:,i]
            differentation_lines.append(np.percentile(LP_score, qr))

        # Store in PI
        self.differentation_lines = differentation_lines

    def _evaluate_benign_samples(self, verbose):
        ''' Private function 
        - Samples are extracted from the test dataset
        total_count   : # of samples extracted from dataset
        correct_count : # of samples (classified correctly)
        valid_count   : # of samples (classified correctly & classified as benign)
        '''

        # Load (test) dataset and model 
        X, Y = self.test_dataset
        model = (self.model).eval()

        # Create intermediate variables 
        num_count, correct_count, valid_count = len(X), len(X), 0
        LPs, LPs_score = [], []

        for i in range(correct_count):
            x, y = X[i], Y[i]
            
            # Filter out samples can not be correctly classified by the given model 
            output = model.forward(torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32)))
            y_ = (output.max(1, keepdim=True)[1]).item() 
            if y_ != y:
                correct_count -= 1
                continue

            # Generate experimental result 
            result, LP_status, LP_risk_score = self.property_match(x, y_, verbose) 

            # Record experimental info 
            LPs.append(LP_status)
            LPs_score.append(LP_risk_score)
            valid_count += result

        if verbose:
            print('Evaluate on benign samples with test set')
            print('# of samples'.ljust(45), ':', num_count)
            print('# of correctly classified samples'.ljust(45), ':', correct_count)
            print('# of correctly classified samples')
            print('     which are indentified as "benign"'.ljust(45), ':', valid_count)
            print('True Negative Rate, TNR (B -> B)'.ljust(45), ':', round((valid_count/correct_count), 3), '(', valid_count, '/', correct_count, ')')

        return num_count, correct_count, valid_count, LPs, LPs_score

    def _evaluate_adversarial_samples(self, verbose):
        ''' Private function 
        - Samples are extracted from the test dataset
        total_count   : # of samples extracted from dataset
        correct_count : # of samples (classified correctly)
        valid_count   : # of samples (classified correctly & classified as benign)
        '''

        # Create attack for generating adversarial samples         
        import attacker
        if self.meta_params['adv_attack'] == 'i_FGSM':
            A = attacker.iterative_FGSM_attacker()
        elif self.meta_params['adv_attack'] == 'JSMA':
            A = attacker.JSMA_attacker()
        elif self.meta_params['adv_attack'] == 'CW_L2':
            A = attacker.CW_L2_attacker()
        else:
            A = NotImplemented

        # Load (test) dataset and model 
        X, Y = self.test_dataset
        model = (self.model).eval()

        # Create intermediate variables 
        LPs, LPs_score = [], []
        num_count, correct_count, valid_count = len(X), len(X), 0
        success_count, non_success_count = 0, 0
        BB_count, BA_count, AA_count, AB_count = 0, 0, 0, 0
        eps, eps_incre_unit, eps_upper_bound = 0, 0.01, 1 

        for i in range(correct_count):
            # Debug information 
            # print('Conduct', i, 'th attack:', self.meta_params['adv_attack'])
            x, y = X[i], Y[i]
            adv_x, adv_y = None, None 

            # Filter out samples can not be correctly classified by the given model 
            output = model.forward(torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32)))
            y_ = (output.max(1, keepdim=True)[1]).item() 
            if y_ != y:
                correct_count -= 1
                continue

            # Iterative attack process start, slightly increase eps until larger than the upper bound 
            is_attack_successful = False
            while (not is_attack_successful):
                eps += eps_incre_unit
                if eps > eps_upper_bound:
                    break  

                adv_x, is_adv_success = A.create_adv_input(x, y, model, eps)
                if is_adv_success:
                    is_attack_successful = True
                    adv_x = (adv_x.detach().numpy())[0]

            output = model.forward(torch.from_numpy(np.expand_dims(adv_x, axis=0).astype(np.float32)))
            adv_y = (output.max(1, keepdim=True)[1]).item()
            result, LP_status, LP_risk_score = self.property_match(adv_x, adv_y, verbose) 
            LPs.append(LP_status)
            LPs_score.append(LP_risk_score)

            # result == 1 -> benign 
            # B 
            if (not is_attack_successful):
                non_success_count += 1
                BB_count += 1 if result==1 else 0 
                BA_count += 1 if result==0 else 0
            # A
            else: 
                success_count += 1
                AB_count += 1 if result==1 else 0
                AA_count += 1 if result==0 else 0
                
        # Record AST 
        assert (success_count+non_success_count)==correct_count
        AST = success_count/correct_count

        if verbose:
            print('Evaluate on adversarial samples with test set')
            print('# of samples'.ljust(45), ':', num_count)
            print('# of correctly classified samples'.ljust(45), ':', correct_count)
            print('# of correctly classified samples')
            print('     which are indentified as "benign"'.ljust(45), ':', non_success_count)
            print('B -> B count'.ljust(45), ':', BB_count)
            print('B -> A count'.ljust(45), ':', BA_count)
            print()
            print('# of correctly classified samples')
            print('     which are indentified as "adversarial"'.ljust(45), ':', success_count)
            print('A -> B count'.ljust(45), ':', AB_count)
            print('A -> A count'.ljust(45), ':', AA_count)
            print()
            print('Attack success rate'.ljust(45), ':', round(AST, 3), '(', success_count, '/', correct_count, ')')
            
        # valid_count = AA_count + BB_count (PENDING)
        valid_count = AA_count + BB_count 
        return num_count, correct_count, valid_count, LPs, LPs_score

