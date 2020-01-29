import numpy as np
import torch
import torch.nn.functional as F
import copy

def return_LP_from_output(h, to_int):
    h_ = (h.clone()).detach().numpy() 
    if to_int:
        h_[h_>0] = True
        h_ = h_.astype(np.int64)
    return list(h_)

def extract_signature_from_CNN(model, x, to_int=False):
    x = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
    singatures = []

    h1 = F.relu(model.conv1(x))
    singatures.append(return_LP_from_output(h1, to_int))

    h2 = F.relu(F.max_pool2d(model.conv2(h1), 2))
    singatures.append(return_LP_from_output(h2, to_int))

    h3 = F.relu(F.max_pool2d(model.conv3(h2), 2))
    singatures.append(return_LP_from_output(h3, to_int))

    h3 = h3.view(-1, 3*3*32)
    h4 = F.relu(model.fc1(h3))
    singatures.append(return_LP_from_output(h4, to_int))

    return singatures 