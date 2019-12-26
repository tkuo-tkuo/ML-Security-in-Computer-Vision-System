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

def property_match(self, x, y, verbose=True):
    Py = self.LPs_set[y]
    LPs = extract_all_LP(self.model, self.meta_params['model_type'], x)

    #############################################
    # Original Method 
    # result == 1 -> the given input is considered as 'benign'
    # result == 0 -> the given input is considered as 'adversarial'
    #############################################
    '''
    LP_status = []
    LP_risk_score = []

    for i in range(len(LPs)):
        LP_i = Py[i]
        p_i = LPs[i]
        
        status = 'adversarial'
        if p_i in LP_i:
            status = 'benign'

        LP_status.append(status)
        if status == 'benign':
            LP_risk_score.append(1)
        else:
            LP_risk_score.append(0)

    
    result = 1
    if 'adversarial' == LP_status[0] and 'adversarial' == LP_status[1]:
        result = 0

    if verbose:
        if result == 1:
            print(LP_status, 'benign')
        else:
            print(LP_status, 'adversarial')

    return (result, LP_status, LP_risk_score)
    '''
    #############################################

    #############################################
    # Experimental: Method 1 & 2
    #############################################
    LP_status = []
    LP_risk_score = []
    differentiation_lines = self.differentation_lines
    for i in range(len(LPs)):
        differentiation_line = differentiation_lines[i]
        LP_i = np.array(Py[i])
        p_i = np.array(LPs[i])

        prob_LP_i = np.sum(LP_i, axis=0) / LP_i.shape[0]
        diff = prob_LP_i - p_i
        abs_diff = np.absolute(diff)
        # abs_diff[abs_diff<0.9] = 0
        risk_score = np.sum(abs_diff)
        
        status = 'adversarial'
        if risk_score < differentiation_line:
            status = 'benign'

        LP_status.append(status)
        LP_risk_score.append(risk_score)

    result = 0
    if 'benign' == LP_status[0] and 'benign' == LP_status[1] and 'benign' == LP_status[2]:
        result = 1    

    '''
    if verbose:
        if result == 1:
            print(LP_status, 'benign')
        else:
            print(LP_status, 'adversarial')
    '''

    return (result, LP_status, LP_risk_score)
    #############################################


    #############################################
    # Experimental: Method 3 & 4
    #############################################
    '''
    LP_status = []
    LP_risk_score = []
    prob_diff_lines = [-800, -600, -150, 1e-4]
    for i in range(len(LPs)):
        prob_diff_line = prob_diff_lines[i]
        LP_i = np.array(Py[i])
        p_i = np.array(LPs[i])

        # compute probability of LP_i
        prob_LP_i = np.sum(LP_i, axis=0) / LP_i.shape[0]

        # To avoid 0 probability in either prob_LP_i or prob_LP_i_0
        prob_LP_i[prob_LP_i==1.0] = 1.0 - (1/(prob_LP_i.shape[0] + 1))
        prob_LP_i[prob_LP_i==0.0] = 0.0 + (1/(prob_LP_i.shape[0] + 1))
        prob_LP_i_0 = 1 - prob_LP_i

        # This section is for Method 4
        # offset = 0.1
        # weights = np.array(prob_LP_i) 
        # weights[weights <= (0.5-offset)] = -1
        # weights[weights >= (0.5+offset)] = -1
        # weights[weights != -1] = 0
        # weights[weights == -1] = 1
        # 

        B_prob = 0
        for i, neuron_activation in enumerate(p_i):
            # This section is for Method 4
            # use weights[i] for Method 4
            #
            if neuron_activation == 1:
                B_prob += math.log(prob_LP_i[i]) * weights[i]
            else:
                B_prob += math.log(prob_LP_i_0[i]) * weights[i]

        status = 'adversarial'
        if B_prob > prob_diff_line:
            status = 'benign'

        LP_status.append(status)
        LP_risk_score.append(B_prob)

    result = 0
    if 'benign' == LP_status[0] and 'benign' == LP_status[1] and 'benign' == LP_status[2]:
        result = 1    

    if verbose:
        if result == 1:
            print(LP_status, 'benign')
        else:
            print(LP_status, 'adversarial')

    return (result, LP_status, LP_risk_score)
    '''
    #############################################
