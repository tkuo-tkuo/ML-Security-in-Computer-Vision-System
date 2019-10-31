# This utils.py extracts the testable functionalities for
# 1. testability
# 2. readability 

import numpy as np
import torch
import torch.nn.functional as F


def extract_subset_certain_classes(X, Y, first_class, second_class, size_of_sub_dataset):
    sub_X = np.array([x for (idx, x) in enumerate(X) if (Y[idx]==first_class or Y[idx]==second_class)])[:size_of_sub_dataset]
    sub_Y = np.array([y for y in Y if (y==first_class or y==second_class)])[:size_of_sub_dataset]
    sub_Y[sub_Y == first_class] = 0
    sub_Y[sub_Y == second_class] = 1
    return (sub_X, sub_Y)

def extract_subset(X, Y, size_of_sub_dataset):
    return (X[:size_of_sub_dataset], Y[:size_of_sub_dataset])

def return_LP_from_output(h):
    h_ = h.detach().numpy()
    h_[h_>0] = True
    h_ = h_.astype(np.int64)
    squ_h_ = (h_).reshape(-1)
    LP = squ_h_.astype(np.int64)
    return list(LP)

def extract_all_LP(model, model_type, x):
    LPs = []

    # Grab the information
    if model_type == 'CNN':
        x = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        h1 = F.relu(model.conv1(x))
        LPs.append(return_LP_from_output(h1))

        h2 = F.relu(F.max_pool2d(model.conv2(h1), 2))
        h2 = F.dropout2d(h2, p=0.1)
        LPs.append(return_LP_from_output(h2))

        h3 = F.relu(F.max_pool2d(model.conv3(h2), 2))
        LPs.append(return_LP_from_output(h3))

        h3 = h3.view(-1, 3*3*32)
        h4 = F.relu(model.fc1(h3))
        LPs.append(return_LP_from_output(h4))

        '''
        x = F.relu(self.conv1(x)) # (24, 24, 16)
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # (20, 20, 16) -> (10, 10, 16)
        x = F.relu(F.max_pool2d(self.conv3(x), 2)) # (6, 6, 32) -> (3, 3, 32)
        x = x.view(-1, 3*3*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        '''
    else:
        x = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        h1 = model.relu(model.layer1(x))
        LPs.append(return_LP_from_output(h1))

        h2 = model.relu(model.layer2(h1))
        LPs.append(return_LP_from_output(h2))

        h3 = model.relu(model.layer3(h2))
        LPs.append(return_LP_from_output(h3))

        h4 = model.relu(model.layer4(h3))
        LPs.append(return_LP_from_output(h4))

    return LPs