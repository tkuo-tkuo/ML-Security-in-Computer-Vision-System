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

def extract_all_LP(model, model_type, x):
    LPs = []

    # Grab the information
    if model_type == 'CNN':
        x = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        x = F.relu(model.conv1(x))
        x = F.relu(F.max_pool2d(model.conv2(x), 2))
        x = F.relu(F.max_pool2d(model.conv3(x), 2))
        x = x.view(-1, 3*3*64)
        h1 = F.relu(model.fc1(x))
        h1[h1>0] = True
        LP = h1.reshape(-1).detach().numpy().astype(np.int64)
        LPs.append(list(LP))
    else:
        x = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
        h1 = model.relu(model.layer1(x))
        h1_ = h1.detach().numpy()
        h1_[h1_>0] = True
        LP = h1_.astype(np.int64)
        LPs.append(list(LP[0]))

        h2 = model.relu(model.layer2(h1))
        h2_ = h2.detach().numpy()
        h2_[h2_ > 0] = True
        LP = h2_.astype(np.int64)
        LPs.append(list(LP[0]))

        h3 = model.relu(model.layer3(h2))
        h3_ = h3.detach().numpy()
        h3_[h3_ > 0] = True
        LP = h3_.astype(np.int64)
        LPs.append(list(LP[0]))

        h4 = model.relu(model.layer4(h3))
        h4_ = h4.detach().numpy()
        h4_[h4_ > 0] = True
        LP = h4_.astype(np.int64)
        LPs.append(list(LP[0]))

    return LPs