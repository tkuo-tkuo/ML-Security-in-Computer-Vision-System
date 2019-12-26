import numpy as np

def extract_subset_certain_classes(X, Y, first_class, second_class, size_of_sub_dataset):
    sub_X = np.array([x for (idx, x) in enumerate(X) if (Y[idx]==first_class or Y[idx]==second_class)])[:size_of_sub_dataset]
    sub_Y = np.array([y for y in Y if (y==first_class or y==second_class)])[:size_of_sub_dataset]
    sub_Y[sub_Y == first_class] = 0
    sub_Y[sub_Y == second_class] = 1
    return (sub_X, sub_Y)

def extract_subset(X, Y, size_of_sub_dataset):
    return (X[:size_of_sub_dataset], Y[:size_of_sub_dataset])
