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

'''
Classicial FGSM (Fast Gradient Sign Method) attack
Category: Untargeted, Gradient-based
'''
class FGSM_attacker():

    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import GradientSignAttack
        adversary = GradientSignAttack(model, loss_fn=nn.CrossEntropyLoss())
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1


'''
Jacobian Saliency Map Attack
Category: L0, Targeted, Gradient-based
'''
class JSMA_attacker():

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
        adversary = JacobianSaliencyMapAttack(model, self.num_classes, loss_fn=nn.CrossEntropyLoss())
        perturbed_data = adversary.perturb(data, target)
        
        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        if final_pred.item() == y:
            return perturbed_data, 0
        else:
            return perturbed_data, 1
        
'''
CarliniWagner L2 Attack
Category: L2, Untargeted, Gradient-based
Illustration: Currently, the most powerful attack among widely-adopted adversarial attacks 
'''
class CW_L2_attacker():
    
    def __init__(self):
        self.num_classes = 10

    def create_adv_input(self, x, y, model):
        # Prepare copied model 
        model = copy.deepcopy(model)

        # Prepare input and corresponding label 
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = torch.from_numpy(np.array([y]).astype(np.int64))
        data.requires_grad = True
        
        from advertorch.attacks import CarliniWagnerL2Attack
        adversary = CarliniWagnerL2Attack(model, self.num_classes, max_iterations=100, abort_early=True)
        perturbed_data = adversary.perturb(data, target)

        # Have to be different
        output = model.forward(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1