import torch 
import torchvision
import torchvision.transforms as transforms

import numpy as np

def create_MNIST_dataset(num_of_train_dataset=50000, num_of_test_dataset=10000, is_flatten=False):
    # create variables for later usage 
    train_dataset, test_dataset = None, None
    train_X, train_Y = None, None 
    test_X, test_Y = None, None 

    # load from torchvision
    print(num_of_train_dataset, num_of_test_dataset)
    MNIST_train_dataset = torchvision.datasets.MNIST(root='./data', transform=transforms.ToTensor(), train=True, download=True)
    MNIST_test_dataset = torchvision.datasets.MNIST(root='./data', transform=transforms.ToTensor(), train=False, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=MNIST_train_dataset, batch_size=num_of_train_dataset, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=MNIST_test_dataset, batch_size=num_of_test_dataset, shuffle=False)

    # create train dataset 
    for (samples, labels) in train_loader:
        if is_flatten: train_X = samples.reshape(-1, 784).numpy()
        else: train_X = samples.numpy()
        train_Ｙ = labels.numpy()
        break

    train_dataset = (train_X, train_Y)

    # create test dataset
    for (samples, labels) in test_loader:
        if is_flatten: test_X = samples.reshape(-1, 784).numpy()
        else: test_X = samples.numpy()
        test_Y = labels.numpy()
        break
        
    test_dataset = (test_X, test_Y) 

    return (train_dataset, test_dataset) 
