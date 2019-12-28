import numpy as np
import copy
import matplotlib.pyplot as plt

class Monitor():

    def __init__(self):
        pass

    def compute_prob(self, PI, verbose=False):
        _, _, (B_LPs, A_LPs), (B_LPs_score, A_LPs_score) = PI.evaluate_algorithm_on_test_set(verbose, on_robustified_model=False, on_twisted_model=True)
        
        BLPs, ALPs = np.array(B_LPs), np.array(A_LPs) 

        BLPs[BLPs=='benign'] = 1
        BLPs[BLPs=='adversarial'] = 0
        BLPs = BLPs.astype(np.int)
        prob_BLPs = np.sum(BLPs, axis=0) / BLPs.shape[0]

        ALPs[ALPs=='benign'] = 1
        ALPs[ALPs=='adversarial'] = 0
        ALPs = ALPs.astype(np.int)
        prob_ALPs = np.sum(ALPs, axis=0) / ALPs.shape[0]

        if verbose:
            print('This indicates the portion of inputs to be judgedprob_ALPs, prob_BLPs, A_LPs_score, B_LPs_score as "benign"')
            print(prob_BLPs, 'test dataset (benign)')
            print(prob_ALPs, 'test dataset (adversarial)')

        return (prob_ALPs, prob_BLPs, A_LPs_score, B_LPs_score)

    def draw_exp_results(self, results, dropout_rate, savefig=False):
        prob_ALPs, prob_BLPs, A_LPs_score, B_LPs_score = results
        
        # Draw the first figure 
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
            plt.savefig('exp13_1_'+str(int(dropout_rate*100)))
        plt.show()

        # Draw the second figure 
        plt.figure(figsize=(18,3))
        B_LPs_score, A_LPs_score = np.array(B_LPs_score), np.array(A_LPs_score)
        for i in range(B_LPs_score.shape[1]):
            B_score, A_score = B_LPs_score[:,i], A_LPs_score[:, i]
            B_indices, A_indices = np.arange(B_score.shape[0]), np.arange(A_score.shape[0])
            plt.subplot(1, 4, i+1)
            plt.plot(B_score, B_indices, 'go')
            plt.plot(A_score, A_indices, 'ro')
            plt.title(str(i+1)+'-th layer (dr='+str(dropout_rate)+')')

        if savefig: 
            plt.savefig('exp13_2_'+str(int(dropout_rate*100)))

        plt.show()
