import unittest
import warnings
import random

import property_inference_interface
from LP_utils import * 

###########################################
# Global variables (control)
###########################################
DEV = False

###########################################
# Global variables (meta parameters)
###########################################
META_PARAMS = {
    'num_of_LPs': 4,
    'size_of_train_set': 100,
    'size_of_test_set': 10,
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
        self.PI = property_inference_interface.PropertyInferenceInterface(META_PARAMS)
        self.PI.load_model(MODEL_PT_FILE)

    '''
    Func 'generate_twisted_model' should have 
    the same accurancy & FNR (B -> B) as original model.
    Here we compare the accurancy. 
    '''
    @unittest.skipIf(DEV, 'Test on DEV unittests only')
    def test_PI_generate_twisted_model_func＿1(self):
        # Generate twisted model 
        self.PI.set_dropout_rate(0)
        self.PI.generate_twisted_model(MODEL_TYPE, 0)

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
        # Generate twisted model 
        self.PI.set_dropout_rate(0)
        self.PI.generate_twisted_model(MODEL_TYPE, 0)

        # Return the original model FNR
        self.PI.generate_LPs()
        (_, original_B_FNR), _, _, _ = self.PI.evaluate_algorithm_on_test_set(verbose=False)
         
        # Return the twisted model FNR 
        import copy
        (_, twisted_B_FNR), _, _, _ = self.PI.evaluate_algorithm_on_test_set(verbose=False, on_twisted_model=True)

        # Check 
        self.assertEqual(original_B_FNR, twisted_B_FNR)        
        
    '''
    Func 'generate_twisted_model' should have 
    the same weights as original model. 
    Here we randomly sample values of weights from both models and compare them.  
    '''
    @unittest.skipIf(DEV, 'Test on DEV unittests only')
    def test_PI_generate_twisted_model_func＿3(self):
        # Generate twisted model 
        self.PI.set_dropout_rate(0)
        self.PI.generate_twisted_model(MODEL_TYPE, 0)

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
    test _create_train_dataset & _create_test_dataset
    '''
    @unittest.skipIf(DEV, 'Test on DEV unittests only')
    def test_PI_prepare_dataset_func(self):
        # sub-unittest 1
        META_PARAMS_copy = copy.deepcopy(META_PARAMS)
        META_PARAMS_copy['size_of_train_set'] = 500
        META_PARAMS_copy['size_of_test_set'] = 50

        PI = property_inference_interface.PropertyInferenceInterface(META_PARAMS_copy)
        (train_X, train_Y) = PI.train_dataset
        self.assertTupleEqual(train_X.shape, (500, 1, 28, 28))
        self.assertTupleEqual(train_Y.shape, (500, ))
        (test_X, test_Y) = PI.test_dataset
        self.assertTupleEqual(test_X.shape, (50, 1, 28, 28))
        self.assertTupleEqual(test_Y.shape, (50, ))

        # sub-unittest 2
        META_PARAMS_copy = copy.deepcopy(META_PARAMS)
        META_PARAMS_copy['size_of_train_set'] = 100
        META_PARAMS_copy['size_of_test_set'] = 10
        META_PARAMS_copy['flatten'] = True

        PI = property_inference_interface.PropertyInferenceInterface(META_PARAMS_copy)
        (train_X, train_Y) = PI.train_dataset
        self.assertTupleEqual(train_X.shape, (100, 1*28*28))
        self.assertTupleEqual(train_Y.shape, (100, ))
        (test_X, test_Y) = PI.test_dataset
        self.assertTupleEqual(test_X.shape, (10, 1*28*28))
        self.assertTupleEqual(test_Y.shape, (10, ))

    '''
    test LP_utils/return_LP_from_output
    We want to ensure return_LP_from_output:
    1. perform correct computation (return correct results)
    2. do not have side-effect 
    '''
    @unittest.skipIf(DEV, 'Test on DEV unittests only')
    def test_LP_utils_return_LP_from_output_func(self):
        # Create a Tensor 
        SHAPE, RETURN_SHAPE = (3, 3, 3, 3), 3*3*3*3
        t1 = torch.rand(SHAPE) + 1e-3
        t2 = torch.zeros(SHAPE)
        t1_, t2_ = t1.clone(), t2.clone() # for side-effect checking

        # Check return 
        r1, r2 = np.array(return_LP_from_output(t1)), np.array(return_LP_from_output(t2))
        oracle_r1 = (torch.ones(RETURN_SHAPE).numpy()).astype(np.int64)
        oracle_r2 = (torch.zeros(RETURN_SHAPE).numpy()).astype(np.int64)
        self.assertTrue(np.array_equal(r1, oracle_r1))
        self.assertTrue(np.array_equal(r2, oracle_r2))

        # Check side-effect 
        self.assertTrue(torch.all(torch.eq(t1, t1_)).item())
        self.assertTrue(torch.all(torch.eq(t2, t2_)).item())

    '''
    test LP_utils/extract_all_LP
    '''
    def test_LP__utils_extract_all_LP(self):
        self.PI.set_dropout_rate(0.1)
        self.PI.generate_twisted_model(MODEL_TYPE, 5)
        self.PI.generate_LPs(on_retrained_model=False, on_twisted_model=True)



if __name__ == '__main__':
    unittest.main()