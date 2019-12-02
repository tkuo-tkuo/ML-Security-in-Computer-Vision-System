import numpy as np

class Monitor():

    def __init__(self):
        pass

    def calculate_stats(self, file_name):
        import numpy as np
        import csv
        
        info = []
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                row = [float(x) for x in row]
                info.append(row)
        csv_file.close()
        
        info = np.array(info)
        shape = info.shape
        (num_of_exp, _) = shape
        
        if num_of_exp > 100:
            info = info[:100]
        
        print(info.shape)
            
        mean = np.mean(info, axis=0)
        std = np.std(info, axis=0)
        mean = np.array([round(x, 4) for x in mean])
        std = np.array([round(x, 3) for x in std])
        
        for i in range(len(mean)):
            print(mean[i], std[i])
        
        return (mean, std)

    def check(self, file_name, auto_update=False):
        import time, threading
        from IPython.display import clear_output
        clear_output()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()+(3600*8))))
        file_name = 'test.csv'
        self.calculate_stats(file_name)
        if auto_update: 
            threading.Timer(10, monitor).start()

    def calculate_stats_iqr(self, file_name):
        import numpy as np
        import csv
        
        info = []
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                row = [float(x) for x in row]
                info.append(row)
        csv_file.close()
        
        info = np.array(info)
        shape = info.shape
        (num_of_exp, _) = shape
        
        if num_of_exp > 100:
            info = info[:100]
        
        mean = np.mean(info, axis=0)
        std = np.std(info, axis=0)
        mean = np.array([round(x, 4) for x in mean])
        std = np.array([round(x, 3) for x in std])

        iqr_infos = []
        for iqr in [0, 0.25, 0.5, 0.75, 1]:
            iqr_info = [round(x, 4) for x in np.quantile(info, iqr, interpolation='midpoint', axis=0)]
            iqr_infos.append(iqr_info)
            
        return iqr_infos

    def draw(self, args):
        iqr_infos_list = []
        for file_name in args['file_names']:
            iqr_infos_list.append(self.calculate_stats_iqr(file_name))
            
        iqr_infos_list = np.array(iqr_infos_list)

        import matplotlib.pyplot as plt
        fig, _ = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(24, 4))
        fig.text(0.5, 0.02, args['x_label'], ha='center')

        '''
        Sub figure 1: FPR
        '''
        plt.subplot(1, 3, 1)
        data = list(iqr_infos_list[:, :, 4])
        data = [(1-x) for x in data]
        plt.boxplot(data, labels=args['sub_x_labels'])
        plt.title('False Positive Rate (FPR)') # Hyperparameters
        plt.ylim(-0.1, 1.1)

        '''
        Sub figure 2: FNR
        '''
        plt.subplot(1, 3, 2)
        data = list(iqr_infos_list[:, :, 5])
        data = [(1-x) for x in data]
        plt.boxplot(data, labels=args['sub_x_labels'])
        plt.title('False Negative Rate (FNR)') # Hyperparameters
        plt.ylim(-0.1, 1.1)

        '''
        Sub figure 3: Average Curve
        '''
        benign_means, benign_stds = [], []
        adv_means, adv_stds = [], []
        for file_name in args['file_names']:
            (mean, std) = self.calculate_stats(file_name)
            benign_means.append(mean[4])
            benign_stds.append(std[4])
            adv_means.append(mean[5])
            adv_stds.append(std[5])    

        ys = [benign_means, adv_means]
        titles = ['Average FPR', 'Average FNR'] # Hyperparameters

        plt.subplot(1, 3, 3)
        x = args['x']
        for idx, (y, title) in enumerate(zip(ys, titles)):
            y = [(1-i) for i in y]
            if idx == 0:
                plt.plot(x, y, 'bo')
            else:
                plt.plot(x, y, 'ro')

        for idx, (y, title) in enumerate(zip(ys, titles)):
            y = [(1-i) for i in y]
            if idx == 0:
                plt.plot(x, y, 'b')
            else:
                plt.plot(x, y, 'r')
            
        plt.legend(titles, loc='upper right')
        # plt.axis(args['axis'])
        plt.ylim(-0.1, 1.1)
        plt.title('Average FPR & FNR') # Hyperparameters
        plt.savefig(args['export_file_name'])

    def robustify_by_dr(self, PI):
        import copy
        import numpy as np
        PI.model = copy.deepcopy(PI.robustified_model)
        PI.generate_LPs()
        (B_detect_ratio, A_detect_ratio), (B_LPs, A_LPs), (B_LPs_score, A_LPs_score) = PI.evaluate_algorithm_on_test_set(verbose=False)
        print(B_detect_ratio, A_detect_ratio)

        qr = '95'
        BLPs, ALPs = np.array(B_LPs), np.array(A_LPs) 
        print(BLPs.shape, ALPs.shape)

        BLPs[BLPs=='benign'] = 1
        BLPs[BLPs=='adversarial'] = 0
        BLPs = BLPs.astype(np.int)
        prob_BLPs = np.sum(BLPs, axis=0) / BLPs.shape[0]

        ALPs[ALPs=='benign'] = 1
        ALPs[ALPs=='adversarial'] = 0
        ALPs = ALPs.astype(np.int)
        prob_ALPs = np.sum(ALPs, axis=0) / ALPs.shape[0]

        print('This indicates the portion of inputs to be judgedprob_ALPs, prob_BLPs, A_LPs_score, B_LPs_score as "benign"')
        print(prob_BLPs, 'test dataset (benign)')
        print(prob_ALPs, 'test dataset (adversarial)')
        return (prob_ALPs, prob_BLPs, A_LPs_score, B_LPs_score)

    def draw_rob_exp_results(self, results, dropout_rate, savefig=False):
        import matplotlib.pyplot as plt
        prob_ALPs, prob_BLPs, A_LPs_score, B_LPs_score = results
        
        # create plot
        num_of_layers = 4
        index = np.arange(num_of_layers)
        bar_width, opacity = 0.2, 0.7

        rects1 = plt.bar(index, prob_BLPs, bar_width, alpha=opacity, color='g', label='Test Ben')
        rects2 = plt.bar(index + bar_width, prob_ALPs, bar_width, alpha=opacity, color='r', label='Test Ben')

        plt.xlabel('I-th layer')
        plt.ylabel('Benign ratio')
        plt.title('Benign ratio in different layers (95 qr)')
        plt.xticks(index + bar_width, ('1', '2', '3', '4'))
        plt.legend()
        plt.tight_layout()
        if savefig: 
            plt.savefig('exp10_1_'+str(int(dropout_rate*100)))
        plt.show()

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
            plt.savefig('exp10_2_'+str(int(dropout_rate*100)))
        plt.show()
