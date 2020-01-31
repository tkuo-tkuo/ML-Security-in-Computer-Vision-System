import math
import copy
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from MNIST_models import *
from dataset_utils import *
from LP_utils import *
import attacker 

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

    def generate_adv_img(self, x, y, model, adv_type):
        if adv_type == 'FGSM': A = attacker.FGSMAttacker()
        elif adv_type == 'JSMA': A = attacker.JSMAAttacker()
        elif adv_type == 'CWL2': A = attacker.CWL2Attacker()
        elif adv_type == 'L1PGD': A = attacker.L1PGDAttack()
        elif adv_type == 'L2PGD': A = attacker.L2PGDAttack()
        elif adv_type == 'LINFPGD': A = attacker.LinfPGDAttack()    
        elif adv_type == 'L2BI': A = attacker.L2BasicIterativeAttack()
        elif adv_type == 'LINFBI': A = attacker.LinfBasicIterativeAttack()
        elif adv_type == 'ENL1': A = attacker.ElasticNetL1Attack()
        elif adv_type == 'DNNL2': A = attacker.DDNL2Attack()
        elif adv_type == 'LBFGS': A = attacker.LBFGSAttack()
        elif adv_type == 'SP': A = attacker.SinglePixelAttack()
        elif adv_type == 'LS': A = attacker.LocalSearchAttack()
        elif adv_type == 'ST': A = attacker.SpatialTransformAttack()
        else: A = NotImplemented

        if A is NotImplemented: return None 

        start = time.clock()
        adv_x, is_att_success = A.create_adv_input(x, y, model)
        end = time.clock()
        # print('time for', adv_type.ljust(8),'sample generation', round(end-start, 3))

        if is_att_success:
            adv_x = (adv_x.detach().numpy())[0]
            return adv_x
        else:
            return None 


    def generate_signatures(self, adv_type=None):
        X, Y = self.train_dataset
        model = copy.deepcopy(self.model)
        model.eval()
        set_of_signatures = []

        for i in range(len(X)):
            print(adv_type, i+1)
            x, y = X[i], Y[i]

            if adv_type is None: 
                singatures = extract_signature_from_CNN(model, x)
                set_of_signatures.append(singatures)
            elif not (adv_type is None): 
                adv_x = self.generate_adv_img(x, y, model, adv_type)

                if adv_x is None: continue

                singatures = extract_signature_from_CNN(model, adv_x)
                set_of_signatures.append(singatures)

        return set_of_signatures
