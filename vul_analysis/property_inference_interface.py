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
from dataset_utils import *
from LP_utils import *

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

    def print_meta_params(self):
        ''' Debug function
        - Print all meta parameters line by line 
        '''
        for key in self.meta_params.keys():
            print((str(key)).ljust(20), ':', self.meta_params[key])

    def print_dataset_shape(self):
        ''' Debug function
        - Print shape of dataset (train & test)
        '''
        print('Train dataset')
        print(self.train_dataset[0].shape, self.train_dataset[1].shape)
        print('Test dataset')
        print(self.test_dataset[0].shape, self.test_dataset[1].shape)
        
    # Debug Information
    def print_some_samples(self):
        ''' Debug function
        - Print shape of dataset (train & test)
        '''
        import math
        import matplotlib.pyplot as plt
        X, _ = self.train_dataset
        for i in range(32):
            data = (X[i]).reshape(28, 28)
            plt.subplot(8, 4, (1+i))
            plt.axis('off')
            plt.imshow(data, cmap='gray')

        plt.show()

    def set_meta_params(self, meta_params):
        ''' Initialization function
        - Store all meta parameters in PI 
        '''
        self.meta_params = meta_params

    def prepare_dataset(self):
        ''' Initialization function
        - Load train and test dataset 
        '''
        self._create_train_dataset()
        self._create_test_dataset()

    def _create_train_dataset(self):
        ''' Private function 
        '''
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
        ''' Private function 
        '''
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
        - Currently,CNN only (FC should be added in the future) 
        - Add a dropout layer & Fine-tune weights  
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

    def generate_robustified_model(self, model_type, num_of_epochs=15, dropout_rate=None):
        ''' 
        - Change the architecture & Train a new model from scratches 
        '''

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
        ''' 
        - Generate the provanence set for each output class
        '''
        X, Y = self.train_dataset

        NUM_MNIST_CLASSES = 10 
        num_output_classes = NUM_MNIST_CLASSES

        LPs_set = []
        for i in range(num_output_classes):
            LPs_set.append([])
        for i in range(num_output_classes):
            for _ in range(self.meta_params['num_of_LPs']):
                LPs_set[i].append([])
            
        for i in range(len(X)):
            x, y = X[i], Y[i]
            LPs = extract_all_LP(self.model, self.meta_params['model_type'], x)
            for i in range(len(LPs)):
                (LPs_set[y])[i].append(LPs[i])

        self.LPs_set = LPs_set

    def property_match(self, x, y, verbose=True):
        ''' 
        - Given a sample and its classified outcome, (x, y)
        - We compare provanence of x (p) to the provanence set of y (P)
        - Then, we compute the risk score & decide whether a given sample, x, is 'benign' or 'adversarial' 
        '''
        # Get the provanence set according to the output class y = model(x)
        PS = self.LPs_set[y]
        # Get the provanence of x 
        ps = extract_all_LP(self.model, self.meta_params['model_type'], x)

        # Create intermediate values 
        LP_status, LP_risk_score = [], []
        differentiation_lines = self.differentation_lines
        
        # Go through layer by layer 
        for i in range(len(PS)):
            differentiation_line, P, p = differentiation_lines[i], np.array(PS[i]), np.array(ps[i])

            # Compute score (the method to compute score can be further adjusted)
            prob_P = np.sum(P, axis=0)/(P.shape[0])
            abs_diff = np.absolute(prob_P-p)
            risk_score = np.sum(abs_diff)

            # If score is lower than differentiation line, then it's 'benign'. 
            # Otherwise, it's 'adversarial'.
            status = 'benign' if risk_score < differentiation_line else 'adversarial'

            LP_status.append(status)
            LP_risk_score.append(risk_score)

        # Decide whether a given sample x is 'benign' or 'adversarial'
        # -> the benign_condition should be further adjusted 
        benign_condition = ('benign' == LP_status[0] and 'benign' == LP_status[1] and 'benign' == LP_status[2])
        is_benign = True if benign_condition else False  

        # Debug information 
        # if is_benign:
        #     print(LP_status, 'benign')
        # else:
        #     print(LP_status, 'adversarial')

        return (is_benign, LP_status, LP_risk_score)

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
            if self.meta_params['is_debug']:
                print('Evaluate', i, 'th benign sample ...')

            x, y = X[i], Y[i]
            
            # Filter out samples can not be correctly classified by the given model 
            output = model.forward(torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32)))
            y_ = (output.max(1, keepdim=True)[1]).item() 
            if y_ != y:
                correct_count -= 1
                continue

            # Generate experimental result 
            is_benign, LP_status, LP_risk_score = self.property_match(x, y_, verbose) 

            # Record experimental info 
            LPs.append(LP_status)
            LPs_score.append(LP_risk_score)
            valid_count += 1 if is_benign else 0 

        if verbose:
            NUM_OF_CHAR_INDENT = 50 
            print('Evaluate on benign samples with test set')
            print('# of samples'.ljust(NUM_OF_CHAR_INDENT), ':', num_count)
            print('# of correctly classified samples'.ljust(NUM_OF_CHAR_INDENT), ':', correct_count)
            print('# of correctly classified samples')
            print('     which are indentified as "benign"'.ljust(NUM_OF_CHAR_INDENT), ':', valid_count)
            print('True Negative Rate, TNR (B -> B)'.ljust(NUM_OF_CHAR_INDENT), ':', round((valid_count/correct_count), 3), '(', valid_count, '/', correct_count, ')')
            print()

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
            if self.meta_params['is_debug']:
                print('Evaluate', i, 'th adversarial sample via', self.meta_params['adv_attack'], '...')

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

                adv_x, is_att_success = A.create_adv_input(x, y, model, eps)
                if is_att_success:
                    is_attack_successful = True
                    adv_x = (adv_x.detach().numpy())[0]

            # Debug information 
            if is_attack_successful:
                adv_x_ = np.expand_dims(adv_x, axis=0).astype(np.float32)
                adv_x_ = torch.from_numpy(adv_x_)
                output_ = model.forward(adv_x_)
                adv_y = (output_.max(1, keepdim=True)[1]).item()
                is_benign, LP_status, LP_risk_score = self.property_match(adv_x, adv_y, verbose) 
            else:
                is_benign, LP_status, LP_risk_score = self.property_match(x, y_, verbose) 

            LPs.append(LP_status)
            LPs_score.append(LP_risk_score)

            # B 
            if (not is_attack_successful):
                non_success_count += 1
                BB_count += 1 if is_benign else 0 
                BA_count += 1 if (not is_benign) else 0
            # A
            else: 
                success_count += 1
                AB_count += 1 if is_benign else 0
                AA_count += 1 if (not is_benign) else 0
                
        # Record AST 
        assert (success_count+non_success_count)==correct_count
        AST = success_count/correct_count

        if verbose:
            NUM_OF_CHAR_INDENT = 50 
            print('Evaluate on adversarial samples with test set')
            print('# of samples'.ljust(NUM_OF_CHAR_INDENT), ':', num_count)
            print('# of correctly classified samples'.ljust(NUM_OF_CHAR_INDENT), ':', correct_count)
            print('# of correctly classified samples')
            print('which are NOT succesfully attacked -> "benign"'.ljust(NUM_OF_CHAR_INDENT), ':', non_success_count)
            print('B -> B count'.ljust(NUM_OF_CHAR_INDENT), ':', BB_count)
            print('B -> A count'.ljust(NUM_OF_CHAR_INDENT), ':', BA_count)
            print('True Negative Rate, TNR (B -> B)'.ljust(NUM_OF_CHAR_INDENT), ':', round((BB_count/non_success_count), 3), '(', BB_count, '/', non_success_count, ')')

            print()
            print('# of correctly classified samples')
            print('which are succesfully attacked -> "adversarial"'.ljust(NUM_OF_CHAR_INDENT), ':', success_count)
            print('A -> B count'.ljust(NUM_OF_CHAR_INDENT), ':', AB_count)
            print('A -> A count'.ljust(NUM_OF_CHAR_INDENT), ':', AA_count)
            print('True Positive Rate, TPR (A -> A)'.ljust(NUM_OF_CHAR_INDENT), ':', round((AA_count/success_count), 3), '(', AA_count, '/', success_count, ')')

            print()
            print('Attack success rate'.ljust(NUM_OF_CHAR_INDENT), ':', round(AST, 3), '(', success_count, '/', correct_count, ')')

        # valid_count = AA_count + BB_count (PENDING)
        valid_count = AA_count + BB_count 
        return num_count, correct_count, valid_count, LPs, LPs_score

