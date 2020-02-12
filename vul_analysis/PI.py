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

import cv2
import pickle

from matplotlib import pyplot as plt

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
        print(self.test_dataset[0].shape)

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

        adv_x, is_att_success = A.create_adv_input(x, y, model)

        if is_att_success:
            adv_x = (adv_x.detach().numpy())[0]
            return adv_x
        else:
            return None 


    def generate_signatures(self, adv_type=None):
        prefixs = ['store_zero/', 'store_one/', 'store_two/', 'store_three/', 'store_four/', 
                'store_five/', 'store_six/', 'store_seven/', 'store_eight/', 'store_nine/']    

        X, Y = self.train_dataset
        model = copy.deepcopy(self.model)
        model.eval()
        set_of_signatures = []

        for i in range(len(X)):
            print(adv_type, i+1)
            x, y = X[i], Y[i]

            if adv_type is None: 
                data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
                output = model(data).detach().numpy()
                prediction = np.argmax(output, axis=1)
  
                if (prediction[0] != y):
                    continue

                singatures = extract_signature_from_CNN(model, x)

                fn = 'store_subs_fadv/'+prefixs[prediction[0]]+'normal_'+str(i+1)+'.txt'
                print(fn)
                with open(fn, "wb") as fp:   
                    pickle.dump(singatures, fp)

            elif not (adv_type is None): 
                adv_x = self.generate_adv_img(x, y, model, adv_type)
                if adv_x is None: continue

                data = torch.from_numpy(np.expand_dims(adv_x, axis=0).astype(np.float32))
                output = model(data).detach().numpy()
                prediction = np.argmax(output, axis=1)
  
                if (prediction[0] == y):
                    continue

                fn = 'store_subs_fadv/'+prefixs[prediction[0]]+adv_type+'_'+str(i+1)+'.txt'
                print(fn)
                singatures = extract_signature_from_CNN(model, adv_x)
                with open(fn, "wb") as fp:   
                    pickle.dump(singatures, fp)

        # return set_of_signatures

    def eval_sub_guards(self, guards, adv_type=None):
        X, Y = self.test_dataset
        model = copy.deepcopy(self.model)
        model.eval()
        total_count = 0 
        correct_count = 0

        for i in range(len(X)):
            x, y = X[i], Y[i]

            if adv_type is None: 
                # Load  
                fn = 'adv_images/'+'benign'+str(i)+'.npy'
                x = np.load(fn)

                data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
                label = torch.from_numpy(np.array([y]).astype(np.int64))
                # Forwarding
                outputs = model.forward(data).detach().numpy()
                prediction = np.argmax(outputs, axis=1)
                is_correct = (prediction == label.numpy()).item()

                # if f(x) == y: 
                if is_correct:
                    # extract singatures
                    singatures = extract_signature_from_CNN(model, x)
                    # select appropriate sub-guard (ground-truth sub-guard)
                    sub_guard = guards[y]

                    # test by sub-guard 
                    f1, f2, f3, f4 = preprocess(singatures)
                    outputs = sub_guard.forward(f1, f2, f3, f4)
                    label = torch.from_numpy(np.array([[1, 0]])).float()
                    prediction = (outputs.max(1, keepdim=True)[1]).item()     
                    if (prediction == 0): 
                        correct_count += 1
                        total_count += 1
                    else:
                        total_count += 1
                else: 
                    continue

            elif not (adv_type is None): 
                # adv_x = self.generate_adv_img(x, y, model, adv_type)
                # if adv_x is None: continue

                # Load (those adv samples successes) 
                fn = 'adv_images/'+adv_type+str(i)+'.npy'
                try: 
                    adv_x = np.load(fn)
                except: 
                    continue

                data = torch.from_numpy(np.expand_dims(adv_x, axis=0).astype(np.float32))
                label = torch.from_numpy(np.array([y]).astype(np.int64))
                # Forwarding
                outputs = model.forward(data).detach().numpy()
                prediction = np.argmax(outputs, axis=1)
                is_correct = (prediction == label.numpy()).item()

                # if f(adv) != y: 
                if not is_correct:
                    # extract singatures
                    singatures = extract_signature_from_CNN(model, adv_x)

                    # sselect appropriate sub-guard (f(adv) sub-guard)
                    class_index = prediction[0]
                    sub_guard = guards[class_index]

                    # test by sub-guard 
                    f1, f2, f3, f4 = preprocess(singatures)
                    outputs = sub_guard.forward(f1, f2, f3, f4)
                    label = torch.from_numpy(np.array([[0, 1]])).float()

                    prediction = (outputs.max(1, keepdim=True)[1]).item()     
                    if (prediction == 1): 
                        correct_count += 1
                        total_count += 1
                    else:
                        total_count += 1

                else: 
                    continue

        print('correct_count:', correct_count, 'total_count:', total_count)
        print('acc:', correct_count/total_count)  

    def eval_guard(self, guard, adv_type=None):
        X, Y = self.test_dataset
        model = copy.deepcopy(self.model)
        model.eval()
        total_count = 0 
        correct_count = 0

        for i in range(len(X)):
            x, y = X[i], Y[i]

            if adv_type is None: 
                # Load  
                fn = 'adv_images/'+'benign'+str(i)+'.npy'
                x = np.load(fn)

                data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
                label = torch.from_numpy(np.array([y]).astype(np.int64))
                # Forwarding
                outputs = model.forward(data).detach().numpy()
                prediction = np.argmax(outputs, axis=1)
                is_correct = (prediction == label.numpy()).item()

                # if f(x) == y: 
                if is_correct:
                    # extract singatures
                    singatures = extract_signature_from_CNN(model, x)

                    # test by sub-guard 
                    f1, f2, f3, f4 = preprocess(singatures)
                    outputs = guard.forward(f1, f2, f3, f4)
                    label = torch.from_numpy(np.array([[1, 0]])).float()
                    prediction = (outputs.max(1, keepdim=True)[1]).item()     
                    if (prediction == 0): 
                        correct_count += 1
                        total_count += 1
                    else:
                        total_count += 1
                else: 
                    continue

            elif not (adv_type is None): 
                # adv_x = self.generate_adv_img(x, y, model, adv_type)
                # if adv_x is None: continue

                # Load (those adv samples successes) 
                fn = 'adv_images/'+adv_type+str(i)+'.npy'
                try: 
                    adv_x = np.load(fn)
                except: 
                    continue

                data = torch.from_numpy(np.expand_dims(adv_x, axis=0).astype(np.float32))
                label = torch.from_numpy(np.array([y]).astype(np.int64))
                # Forwarding
                outputs = model.forward(data).detach().numpy()
                prediction = np.argmax(outputs, axis=1)
                is_correct = (prediction == label.numpy()).item()

                # if f(adv) != y: 
                if not is_correct:
                    # extract singatures
                    singatures = extract_signature_from_CNN(model, adv_x)

                    # test by sub-guard 
                    f1, f2, f3, f4 = preprocess(singatures)
                    outputs = guard.forward(f1, f2, f3, f4)
                    label = torch.from_numpy(np.array([[0, 1]])).float()

                    prediction = (outputs.max(1, keepdim=True)[1]).item()     
                    if (prediction == 1): 
                        correct_count += 1
                        total_count += 1
                    else:
                        total_count += 1

                else: 
                    continue

        print('correct_count:', correct_count, 'total_count:', total_count)
        print('acc:', correct_count/total_count)  