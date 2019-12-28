import unittest
import warnings
import random

import property_inference_interface

###########################################
# Global variables (control)
###########################################
DEV = False

###########################################
# Global variables (meta parameters)
###########################################
META_PARAMS = {
    'num_of_LPs': 4,
    'size_of_train_set': 1000,
    'size_of_test_set': 100,
    'flatten': False, 
    'model_type': 'CNN',
    'adv_attack': 'i_FGSM',
    'is_debug': False
}
MODEL_PT_FILE = 'MNIST_CNN.pt'
MODEL_TYPE = 'CNN'

# This class be further separated into small test suites if needed
class TestPI(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore')
        self.PI = property_inference_interface.PropertyInferenceInterface()
        self.PI.set_meta_params(META_PARAMS)
        self.PI.prepare_dataset()
        self.PI.load_model(MODEL_PT_FILE)

    '''
    Func 'generate_twisted_model' should have 
    the same accurancy & FNR (B -> B) as original model.
    Here we compare the accurancy. 
    '''
    @unittest.skipIf(DEV, 'Test on DEV unittests only')
    def test_PI_generate_twisted_model_func＿1(self):
        NUM_OF_TWIST_MODEL_TRAIN = 0 

        # Generate twisted model 
        self.PI.generate_twisted_model(MODEL_TYPE, NUM_OF_TWIST_MODEL_TRAIN, dropout_rate=0)

        # Return the original model accurancy
        original_train_acc = self.PI.eval_model('train')
        original_test_acc = self.PI.eval_model('test')
        
        # Return the twisted model accurancy 
        twisted_train_acc = self.PI.eval_model('train', on_twisted_model=True)
        twisted_test_acc = self.PI.eval_model('test', on_twisted_model=True)
        
        # Check 
        self.assertEqual(original_train_acc, twisted_train_acc)
        self.assertEqual(original_test_acc, twisted_test_acc)

    '''
    Func 'generate_twisted_model' should have 
    the same accurancy & FNR (B -> B) as original model.
    Here we compare the FNR. 
    '''
    @unittest.skipIf(DEV, 'Test on DEV unittests only')
    def test_PI_generate_twisted_model_func＿2(self):
        NUM_OF_TWIST_MODEL_TRAIN = 0 

        # Generate twisted model 
        self.PI.generate_LPs()
        self.PI.generate_twisted_model(MODEL_TYPE, NUM_OF_TWIST_MODEL_TRAIN, dropout_rate=0)

        # Return the original model FNR
        (_, original_B_FNR), _, _, _ = self.PI.evaluate_algorithm_on_test_set(verbose=False)
         
        # Return the twisted model FNR 
        import copy
        self.PI.model = copy.deepcopy(self.PI.twisted_model)
        (_, twisted_B_FNR), _, _, _ = self.PI.evaluate_algorithm_on_test_set(verbose=False)

        # Check 
        self.assertEqual(original_B_FNR, twisted_B_FNR)        
        
    '''
    Func 'generate_twisted_model' should have 
    the same weights as original model. 
    Here we randomly sample values of weights from both models and compare them.  
    '''
    @unittest.skipIf(DEV, 'Test on DEV unittests only')
    def test_PI_generate_twisted_model_func＿3(self):
        NUM_OF_TWIST_MODEL_TRAIN = 0 

        # Generate twisted model 
        self.PI.generate_LPs()
        self.PI.generate_twisted_model(MODEL_TYPE, NUM_OF_TWIST_MODEL_TRAIN, dropout_rate=0)

        # Return weights 
        import copy
        original_model, twisted_model = self.PI.model, self.PI.twisted_model 
        original_layers, twisted_layers = copy.deepcopy(original_model.state_dict()), copy.deepcopy(twisted_model.state_dict())

        for (_, original_weights), (_, twisted_weights) in zip(original_layers.items(), twisted_layers.items()):
            o = original_weights.reshape(-1)
            t = twisted_weights.reshape(-1)
            assert len(o) == len(t)
            
            random_ind = random.randint(0, len(o)-1)

            # Check 
            self.assertEqual(o[random_ind], t[random_ind])

    '''
    test _create_train_dataset/_create_test_dataset
    '''
    def test_PI_prepare_dataset_func(self):
        pass 

if __name__ == '__main__':
    unittest.main()