import numpy as np
import torch
import torch.nn.functional as F
import copy

def return_LP_from_output(h):
    h_ = (h.clone()).detach().numpy() 
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
        LPs.append(return_LP_from_output(h2))

        h3 = F.relu(F.max_pool2d(model.conv3(h2), 2))
        LPs.append(return_LP_from_output(h3))

        h3 = h3.view(-1, 3*3*32)
        h4 = F.relu(model.fc1(h3))
        LPs.append(return_LP_from_output(h4))
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