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

class iterative_FGSM_attacker():

    def fgsm_attack(self, image, epsilon, data_grad):
        
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + (epsilon * sign_data_grad)
        # Return the perturbed image
        return perturbed_image

    def create_adv_input(self, x, y, model, epsilon):
        data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        target = np.array([y]).astype(np.int64)
        target = torch.from_numpy(target)

        data.requires_grad = True
        output = model.forward(data)

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
        

        # Have to be different
        output = model.forward(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item():
            return perturbed_data, 0
        else:
            return perturbed_data, 1