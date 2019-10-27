# This discarded_utils.py stores the currently unused functionalities

def extract_subset_certain_ratio(X, Y, ratio):
    import math
    new_train_dataset_size = math.floor(len(X) * ratio)
    sub_X, sub_Y = X[:new_train_dataset_size], Y[:new_train_dataset_size] 
    return (sub_X, sub_Y)

def generate_new_datas(args):
    '''
    X, Y = self.train_dataset
    model = self.model
    is_random_perturbed_inputs_included = self.meta_params['is_ran_per_included']
    random_distribution = self.meta_params['ran_dist']

    first_F, second_F = [[], [], [], []], [[], [], [], []]
    for i in range(len(X)):
        x, y = X[i], Y[i]
        LPs = self.extract_all_LP(x)
        if y == 0:
            for i in range(len(LPs)):
                first_F[i].append(LPs[i])
        else: 
            for i in range(len(LPs)):
                second_F[i].append(LPs[i])

    if is_random_perturbed_inputs_included: 
        # Uniform 
        if random_distribution == 'uniform':
            uniform_dis_range = self.meta_params['uni_range']
            for i in range(len(X)):
                for _ in range(5):
                    x = X[i]
                    random_x = x + ((np.random.random((784)) * uniform_dis_range) - (uniform_dis_range/2))
                    random_x[random_x >= 1] = 1.0
                    random_x[random_x <= 0] = 0.0
                    output = model.forward(torch.from_numpy(np.expand_dims(random_x, axis=0).astype(np.float32)))
                    y = (output.max(1, keepdim=True)[1]).item() # get the index of the max log-probability
                    precondition = self.extract_precondition(x)
                    if y == 0:
                        for i in range(len(precondition)):
                            first_F[i].append(precondition[i])
                    else: 
                        for i in range(len(precondition)):
                            second_F[i].append(precondition[i])

        # Normal 
        elif random_distribution == 'normal':
            normal_std = self.meta_params['normal_std']
            for i in range(len(X)):
                for _ in range(5):
                    x = X[i]
                    random_x = x + np.random.normal(0, normal_std, 784)
                    random_x[random_x >= 1] = 1.0
                    random_x[random_x <= 0] = 0.0
                    output = model.forward(torch.from_numpy(np.expand_dims(random_x, axis=0).astype(np.float32)))
                    y = (output.max(1, keepdim=True)[1]).item() # get the index of the max log-probability
                    precondition = self.extract_precondition(x)
                    if y == 0:
                        for i in range(len(precondition)):
                            first_F[i].append(precondition[i])
                    else: 
                        for i in range(len(precondition)):
                            second_F[i].append(precondition[i])
    '''

    return None 