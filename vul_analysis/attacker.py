import copy, random

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

class FGSMAttacker():
    ''' Official documentation
    classadvertorch.attacks.GradientSignAttack(predict, loss_fn=None, eps=0.3, clip_min=0.0, 
    clip_max=1.0, targeted=False)
    '''

    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import GradientSignAttack
        adversary = GradientSignAttack(model.forward)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1

class JSMAAttacker():
    ''' Official documentation
    classadvertorch.attacks.JacobianSaliencyMapAttack(predict, num_classes, clip_min=0.0, clip_max=1.0, 
    loss_fn=None, theta=1.0, gamma=1.0, comply_cleverhans=False)
    '''

    def __init__(self):
        self.num_classes = 10

    def create_adv_input(self, x, y, model):
        model = copy.deepcopy(model)

        target_y = random.randint(0, 9)
        while y == target_y:
            target_y = random.randint(0, 9)
        
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = np.array([target_y]).astype(np.int64)
        target = torch.from_numpy(target)
        data.requires_grad = True
        
        from advertorch.attacks import JacobianSaliencyMapAttack
        adversary = JacobianSaliencyMapAttack(model.forward, self.num_classes)
        perturbed_data = adversary.perturb(data, target)
        
        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        if final_pred.item() == y:
            return perturbed_data, 0
        else:
            return perturbed_data, 1
        
class CWL2Attacker():
    ''' Official documentation
    classadvertorch.attacks.CarliniWagnerL2Attack(predict, num_classes, confidence=0, targeted=False, 
    learning_rate=0.01, binary_search_steps=9, max_iterations=10000, abort_early=True, initial_const=0.001, 
    clip_min=0.0, clip_max=1.0, loss_fn=None)
    '''
    
    def __init__(self):
        self.num_classes = 10
        self.max_iterations = 200

    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import CarliniWagnerL2Attack
        adversary = CarliniWagnerL2Attack(model.forward, self.num_classes, max_iterations=self.max_iterations)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1

class L1PGDAttack():
    ''' Official documentation
    classadvertorch.attacks.L1PGDAttack(predict, loss_fn=None, eps=10.0, nb_iter=40, eps_iter=0.01, 
    rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    '''
    def __init__(self):
        self.eps = 10.0
        self.nb_iter = 300
        self.eps_iter = 0.3
    
    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import L1PGDAttack
        adversary = L1PGDAttack(model.forward, eps=self.eps, nb_iter=self.nb_iter, eps_iter=self.eps_iter)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1

class L2PGDAttack():
    ''' Official documentation
    classadvertorch.attacks.L2PGDAttack(predict, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.01, 
    rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)[source]
    '''
    def __init__(self):
        self.eps = 0.5
        self.nb_iter = 300
        self.eps_iter = 0.3
    
    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import L2PGDAttack
        adversary = L2PGDAttack(model.forward, eps=self.eps, nb_iter=self.nb_iter, eps_iter=self.eps_iter)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1

class LinfPGDAttack():
    ''' Official documentation
    classadvertorch.attacks.LinfPGDAttack(predict, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.01, 
    rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    '''
    
    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import LinfPGDAttack
        adversary = LinfPGDAttack(model.forward)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1

class L2BasicIterativeAttack():
    ''' Official documentation
    classadvertorch.attacks.L2BasicIterativeAttack(predict, loss_fn=None, eps=0.1, nb_iter=10, 
    eps_iter=0.05, clip_min=0.0, clip_max=1.0, targeted=False)
    '''
    def __init__(self):
        self.eps = 0.5
        self.nb_iter = 100
        self.eps_iter = 0.3
    
    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import L2BasicIterativeAttack
        adversary = L2BasicIterativeAttack(model.forward, eps=self.eps, nb_iter=self.nb_iter, eps_iter=self.eps_iter)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1

class LinfBasicIterativeAttack():
    ''' Official documentation
    classadvertorch.attacks.LinfBasicIterativeAttack(predict, loss_fn=None, eps=0.1, nb_iter=10, eps_iter=0.05, 
    clip_min=0.0, clip_max=1.0, targeted=False)
    '''
    def __init__(self):
        self.eps = 0.3
        self.nb_iter = 30
    
    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import LinfBasicIterativeAttack
        adversary = LinfBasicIterativeAttack(model.forward, eps=self.eps, nb_iter=self.nb_iter)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1

class ElasticNetL1Attack():
    ''' Official documentation
    classadvertorch.attacks.ElasticNetL1Attack(predict, num_classes, confidence=0, targeted=False, 
    learning_rate=0.01, binary_search_steps=9, max_iterations=10000, abort_early=False, initial_const=0.001, 
    clip_min=0.0, clip_max=1.0, beta=0.01, decision_rule='EN', loss_fn=None)
    '''
    
    def __init__(self):
        self.num_classes = 10
        self.max_iterations = 200

    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import ElasticNetL1Attack
        adversary = ElasticNetL1Attack(model.forward, self.num_classes, max_iterations=self.max_iterations)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1

class DDNL2Attack():
    ''' Official documentation
    classadvertorch.attacks.DDNL2Attack(predict, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True, 
    levels=256, clip_min=0.0, clip_max=1.0, targeted=False, loss_fn=None)[source]
    '''
    def __init__(self):
        self.nb_iter = 500

    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import DDNL2Attack
        adversary = DDNL2Attack(model.forward, nb_iter=self.nb_iter)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1

class LBFGSAttack():
    ''' Official documentation
    classadvertorch.attacks.LBFGSAttack(predict, num_classes, batch_size=1, binary_search_steps=9, 
    max_iterations=100, initial_const=0.01, clip_min=0, clip_max=1, loss_fn=None, targeted=False)
    '''

    def __init__(self):
        self.num_classes = 10
        self.max_iterations = 500
        
    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import LBFGSAttack
        adversary = LBFGSAttack(model.forward, self.num_classes, max_iterations=self.max_iterations)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1

class SinglePixelAttack():
    ''' Official documentation
    classadvertorch.attacks.SinglePixelAttack(predict, max_pixels=100, clip_min=0.0, loss_fn=None, 
    clip_max=1.0, comply_with_foolbox=False, targeted=False)
    '''
    def __init__(self):
        self.max_pixels = 250
        
    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import SinglePixelAttack
        adversary = SinglePixelAttack(model.forward, max_pixels=self.max_pixels)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1

class LocalSearchAttack():
    ''' Official documentation
    classadvertorch.attacks.LocalSearchAttack(predict, clip_min=0.0, clip_max=1.0, p=1.0, 
    r=1.5, loss_fn=None, d=5, t=5, k=1, round_ub=10, seed_ratio=0.1, max_nb_seeds=128, 
    comply_with_foolbox=False, targeted=False)
    '''

    def __init__(self):
        self.round_ub = 100
        
    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import LocalSearchAttack
        adversary = LocalSearchAttack(model.forward, round_ub=self.round_ub)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1

class SpatialTransformAttack():
    ''' Official documentation
    classadvertorch.attacks.SpatialTransformAttack(predict, num_classes, confidence=0, 
    initial_const=1, max_iterations=1000, search_steps=1, loss_fn=None, clip_min=0.0, 
    clip_max=1.0, abort_early=True, targeted=False)
    '''

    def __init__(self):
        self.num_classes = 10
        self.max_iterations = 200
        
    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import SpatialTransformAttack
        adversary = SpatialTransformAttack(model.forward, self.num_classes, max_iterations=self.max_iterations)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1
