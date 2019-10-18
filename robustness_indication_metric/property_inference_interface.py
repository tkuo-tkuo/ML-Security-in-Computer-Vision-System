import tensorflow as tf
import keras 
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class NaiveC(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 64)
        self.layer2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.layer1(x))
        return self.layer2(output)

class NormalC(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 64)
        self.layer2 = nn.Linear(64, 10)
        self.layer3 = nn.Linear(10, 2)
        # self.layer3 = nn.Linear(32, 20)
        # self.layer4 = nn.Linear(20, 10)
        # self.layer5 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.layer1(x))
        h2 = self.relu(self.layer2(h1))
        # h3 = self.relu(self.layer3(h2))
        # h4 = self.relu(self.layer4(h3))
        # return self.layer5(h4)
        return self.layer3(h2)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(3*3*32, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # (24, 24, 16)
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # (20, 20, 16) -> (10, 10, 16)
        x = F.relu(F.max_pool2d(self.conv3(x), 2)) # (6, 6, 32) -> (3, 3, 32)
        x = x.view(-1, 3*3*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

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

        self.first_F, self.second_F = None, None
        self.uniq_first_F, self.uniq_second_F = None, None
        self.property_set = None        

    def set_meta_params(self, meta_params):
        self.meta_params = meta_params

    # Debug Information
    def print_meta_params(self):
        for key in self.meta_params.keys():
            print((str(key)).ljust(20), ':', self.meta_params[key])

    def prepare_dataset(self):
        self._create_train_dataset()
        self._create_test_dataset()

    def _create_train_dataset(self):
        X, Y = None, None 
        first_class, second_class = self.meta_params['first_class'], self.meta_params['second_class']
        size_of_dataset = self.meta_params['size_of_train_set']        
        
        for (samples, labels) in self.train_loader:
            if self.meta_params['flatten'] is True:
                X = samples.reshape(-1, 784).numpy()
            else:
                # X = samples.reshape(-1, 28, 28, 1).numpy()
                X = samples.numpy()
            Y = (labels.numpy())
        
            X = np.array([x for (idx, x) in enumerate(X) if (Y[idx]==first_class or Y[idx]==second_class)])[:size_of_dataset]
            Y = np.array([y for y in Y if (y==first_class or y==second_class)])[:size_of_dataset]
            Y[Y == first_class] = 0
            Y[Y == second_class] = 1
            break
        
        self.train_dataset = (X, Y)

    def _create_test_dataset(self):
        X, Y = None, None 
        first_class, second_class = self.meta_params['first_class'], self.meta_params['second_class']
        size_of_dataset = self.meta_params['size_of_test_set']

        for (samples, labels) in self.test_loader:
            if self.meta_params['flatten'] is True:
                X = samples.reshape(-1, 784).numpy()
            else:
                # X = samples.reshape(-1, 28, 28, 1).numpy()
                X = samples.numpy()
            Y = (labels.numpy())
        
            X = np.array([x for (idx, x) in enumerate(X) if (Y[idx]==first_class or Y[idx]==second_class)])[:size_of_dataset]
            Y = np.array([y for y in Y if (y==first_class or y==second_class)])[:size_of_dataset]
            Y[Y == first_class] = 0
            Y[Y == second_class] = 1
            break
        
        self.test_dataset = (X, Y)

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

    def generate_model(self, num_of_epochs=15):
        if self.meta_params['model_type'] == 'naive':
            model = NaiveC()
        elif self.meta_params['model_type'] == 'normal':
            model = NormalC()
        elif self.meta_params['model_type'] == 'CNN':
            model = CNN()

        X, Y = self.train_dataset

        # Optimizer parameters
        loss_func = nn.CrossEntropyLoss()
        lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Training
        for epoch in range(num_of_epochs):
            for idx, data in enumerate(X):

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

    def eval_model(self):
        X, Y = self.train_dataset
        model = self.model

        datas = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(Y.astype(np.int64))
        
        # Forwarding
        outputs = model.forward(datas).detach().numpy()
        predictions = np.argmax(outputs, axis=1)

        total = labels.shape[0]
        correct = (predictions == labels.numpy()).sum().item()
        acc = correct/total
            
        print('Model (train) accurancy:', acc)
        return acc

    def extract_precondition(self, x, y):
        model = self.model 

        # Grab the information
        if self.meta_params['model_type'] == 'CNN':
            x = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
            x = F.relu(model.conv1(x))
            x = F.relu(F.max_pool2d(model.conv2(x), 2))
            x = F.relu(F.max_pool2d(model.conv3(x), 2))
            x = x.view(-1, 3*3*64)
            h1 = F.relu(model.fc1(x))
            h1[h1>0] = True
            precondition = h1.reshape(-1).detach().numpy().astype(np.int64)
            set_of_precondition = list(precondition)
        else:
            x = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
            h1 = model.relu(model.layer1(x))
            h1[h1>0] = True
            precondition = h1.detach().numpy().astype(np.int64)
            set_of_precondition = list(precondition[0])

        return set_of_precondition

    def generate_set_of_preconditions(self):
        X, Y = self.train_dataset
        model = self.model
        is_random_perturbed_inputs_included = self.meta_params['is_ran_per_included']
        random_distribution = self.meta_params['ran_dist']

        # Subsample the training dataset if indicated 
        if not (self.meta_params['subsample_ratio'] is None):
            import math
            ratio = self.meta_params['subsample_ratio']
            new_train_dataset_size = math.floor(len(X) * ratio)
            X, Y = X[:new_train_dataset_size], Y[:new_train_dataset_size] 

        first_F, second_F = [], []
        for i in range(len(X)):
            x, y = X[i], Y[i]
            precondition = self.extract_precondition(x, y)
            if y == 0:
                first_F.append(precondition)
            else: 
                second_F.append(precondition)

        if is_random_perturbed_inputs_included: 
            # Uniform 
            if random_distribution == 'uniform':
                uniform_dis_range = self.meta_params['uni_range']
                for i in range(len(X)):
                    for _ in range(5):
                        x, y = X[i], Y[i]
                        random_x = x + ((np.random.random((784)) * uniform_dis_range) - (uniform_dis_range/2))
                        random_x[random_x >= 1] = 1.0
                        random_x[random_x <= 0] = 0.0
                        output = model.forward(torch.from_numpy(np.expand_dims(random_x, axis=0).astype(np.float32)))
                        y = (output.max(1, keepdim=True)[1]).item() # get the index of the max log-probability
                        precondition = self.extract_precondition(random_x, y)
                        if y == 0:
                            first_F.append(precondition)
                        else: 
                            second_F.append(precondition)

            # Normal 
            elif random_distribution == 'normal':
                normal_std = self.meta_params['normal_std']
                for i in range(len(X)):
                    for _ in range(5):
                        x, y = X[i], Y[i]
                        random_x = x + np.random.normal(0, normal_std, 784)
                        random_x[random_x >= 1] = 1.0
                        random_x[random_x <= 0] = 0.0
                        output = model.forward(torch.from_numpy(np.expand_dims(random_x, axis=0).astype(np.float32)))
                        y = (output.max(1, keepdim=True)[1]).item() # get the index of the max log-probability
                        precondition = self.extract_precondition(random_x, y)
                        if y == 0:
                            first_F.append(precondition)
                        else: 
                            second_F.append(precondition)
                pass 
            else: 
                pass 
        
        uniq_first_F, uniq_second_F = [], []
        _ = [uniq_first_F.append(x) for x in first_F if x not in uniq_first_F]
        _ = [uniq_second_F.append(x) for x in second_F if x not in uniq_second_F]

        self.first_F, self.second_F = first_F, second_F
        self.uniq_first_F, self.uniq_second_F = uniq_first_F, uniq_second_F
        self.property_set = [uniq_first_F, uniq_second_F]

    def print_set_of_preconditions(self):
        print('Length of each precondition (provenance):', len(self.first_F[0]))
        print('Total Input Properties extracted for the first class:', len(self.first_F))
        print('Total Input Properties extracted for the second class:', len(self.second_F))
        print('Total Input Properties extracted for the first class (unique):', len(self.uniq_first_F))
        print('Total Input Properties extracted for the second class (unique):', len(self.uniq_second_F))

    def info_about_set_of_preconditions(self):
        return (len(self.first_F), len(self.second_F), len(self.uniq_first_F), len(self.uniq_second_F))

    def property_match(self, x, y):
        Py = self.property_set[y]
        precondition = self.extract_precondition(x, y)

        result = 0
        if precondition in Py:
            result = 1
        
        return result

    def evaluate_algorithm_on_test_set(self, verbose=True):
        self._double_check_on_train_set()
        benign_detect_ratio = self._evaluate_benign_samples(verbose)
        adversarial_detect_ratio = self._evaluate_adversarial_samples(verbose)
        return (benign_detect_ratio, adversarial_detect_ratio)

    def _double_check_on_train_set(self):
        X, Y = self.train_dataset
        num_of_count = len(X)
        valid_count = 0 
        for i in range(len(X)):
            x, y = X[i], Y[i]
            result = self.property_match(x, y)
            valid_count += result

        # assert valid_count == num_of_count
        # print('Evaluate on benign samples within train set (should acheieve 100%)')
        # print(valid_count, num_of_count, valid_count/num_of_count)

    def _evaluate_benign_samples(self, verbose):
        test_X, test_Y = self.test_dataset
        num_of_count = len(test_X)
        valid_count = 0 
        for i in range(num_of_count):
            x, y = test_X[i], test_Y[i]
            result = self.property_match(x, y)
            valid_count += result

        if verbose:
            print('Evaluate on benign samples with test set')
            print(valid_count, num_of_count, (valid_count/num_of_count))
        return (valid_count/num_of_count)

    def _evaluate_adversarial_samples(self, verbose):
        import attacker
        A = attacker.iterative_FGSM_attacker()
        test_X, test_Y = self.test_dataset
        model = self.model

        num_of_count = len(test_X)
        valid_count = 0        
        success_count = 0
        for i in range(num_of_count):
            print(i)
            x, y = test_X[i], test_Y[i]

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
                    
            result = self.property_match(adv_x, y)
            valid_count += result
        
        assert success_count == num_of_count 
        if verbose:
            print('Evaluate on adversarial samples with test set')
            print((num_of_count - valid_count), num_of_count, (num_of_count - valid_count)/num_of_count)
        return ((num_of_count - valid_count)/num_of_count)