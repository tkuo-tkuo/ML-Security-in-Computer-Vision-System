import math
import copy
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

# from binary_MNIST_models import NaiveC, NormalC, CNN
from MNIST_models import *
from dataset_utils import *
from LP_utils import *

import time

class PIInterface():

    def __init__(self, meta_params):
        self.model = None

        # Load meat parameters
        self.meta_params = meta_params

        # create dataset
        self.train_dataset, self.test_dataset = create_MNIST_dataset(
            self.meta_params['num_of_train_dataset'], 
            self.meta_params['num_of_test_dataset'], 
            self.meta_params['is_flatten'])

        self.LPs_set = None
        self.differentation_lines = None

    def train_model(self, model, loss_func, opt, num_of_epochs=15):
        return train(self.train_dataset, model, loss_func, opt, num_of_epochs)

    def set_model(self, model):
        self.model = model

    def eval_model(self, dataset_type):
        if dataset_type == 'train': X, Y = self.train_dataset
        else: X, Y = self.test_dataset

        return eval_model(self.model, (X, Y))

    def generate_signatures(self, is_benign=True, type_of_attack=None):
        X, Y = self.train_dataset
        model = copy.deepcopy(self.model)

        if not is_benign:
            import attacker
            if type_of_attack == 'FGSM': A = attacker.FGSM_attacker()
            elif type_of_attack == 'JSMA': A = attacker.JSMA_attacker()
            elif type_of_attack == 'CW_L2': A = attacker.CW_L2_attacker()
            else: A = NotImplemented

        model.eval()
        set_of_signatures = []

        for i in range(len(X)):
            print(type_of_attack, i+1)
            x, y = X[i], Y[i]

            if not is_benign:
                start = time.clock()
                adv_x, is_att_success = A.create_adv_input(x, y, model)
                end = time.clock()
                print('time for adv sample generation',(end-start))

                if is_att_success:
                    adv_x = (adv_x.detach().numpy())[0]
                    singatures = extract_signature_from_CNN(model, adv_x)
                else:
                    continue
            else:
                singatures = extract_signature_from_CNN(model, x)

            set_of_signatures.append(singatures)

        return set_of_signatures
