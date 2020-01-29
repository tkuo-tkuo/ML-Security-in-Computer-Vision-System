import math
import matplotlib.pyplot as plt
import numpy as np

def print_database_samples(dataset, num_of_print_samples):
    ''' 
    - Display samples from a given dataset 
    - User can govern the number of samples displayed 
    - Currently, upper limit is 32 samples 
    '''
    X, _ = dataset
    for i in range(num_of_print_samples):
        data = (X[i]).reshape(28, 28)
        plt.subplot(8, 4, (1+i))
        plt.axis('off')
        plt.imshow(data, cmap='gray')

    plt.show()

def print_dataset_shape(dataset):
    ''' 
    - Print the shape of dataset 
    '''
    print('Dataset shape')
    print(dataset[0].shape, dataset[1].shape)


def draw_exp_results(results, dropout_rate, savefig=False, first_fig_prefix_str='', second_fig_prefix_str=''):
    prob_ALPs, prob_BLPs, A_LPs_score, B_LPs_score = results
    
    # Draw the first figure 
    plt.figure(figsize=(12,2))
    num_of_layers, bar_width, opacity = 4, 0.2, 0.7
    index = np.arange(num_of_layers)
    plt.bar(index, prob_BLPs, bar_width, alpha=opacity, color='g', label='Test Ben')
    plt.bar(index + bar_width, prob_ALPs, bar_width, alpha=opacity, color='r', label='Test Adv')

    plt.xlabel('I-th layer')
    plt.ylabel('Benign ratio')
    plt.title('Benign ratio in different layers (95 qr)')
    plt.xticks(index + bar_width, ('1', '2', '3', '4'))
    plt.legend()
    plt.tight_layout()
    if savefig: 
        plt.savefig(first_fig_prefix_str+str(int(dropout_rate*100)))
    plt.show()

    # Draw the second figure 
    plt.figure(figsize=(12,2))
    B_LPs_score, A_LPs_score = np.array(B_LPs_score), np.array(A_LPs_score)
    for i in range(B_LPs_score.shape[1]):
        B_score, A_score = B_LPs_score[:,i], A_LPs_score[:, i]
        B_indices, A_indices = np.arange(B_score.shape[0]), np.arange(A_score.shape[0])
        plt.subplot(1, 4, i+1)
        plt.plot(B_score, B_indices, 'go')
        plt.plot(A_score, A_indices, 'ro')
        plt.title(str(i+1)+'-th layer (dr='+str(dropout_rate)+')')

    if savefig: 
        plt.savefig(second_fig_prefix_str+str(int(dropout_rate*100)))

    plt.show()