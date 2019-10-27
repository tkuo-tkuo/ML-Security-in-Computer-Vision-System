### RQ1: Can we design a metric to approximate the adversarial robustness efficiently?  

#### _Approach 1: Directly estimate the training dataset_ 

There are related papers, which aim to define (theoretical) metrix that highly correlated to robustness. I place description of relevant work in Appendix 1.1. 

**Notations** <br />

  - The metric for adversarial robustness approximation is denoted as ***static estimation of adversarial risk***, <img src="README_images/sta_adv_r_est_formula.png" align="center" border="0" alt="sta\_adv\_r = static\_adv\_rob\_estimation\big(S, f\big) " width="369" height="21" />, where <img src="README_images/S.png" align="center" border="0" alt="S" width="17" height="15" /> and <img src="README_images/f.png" align="center" border="0" alt="f" width="12" height="19" /> indicate the training dataset and neural network trained. 
  - For each pair of machine learning models (<img src="README_images/f_A.png" align="center" border="0" alt=" f_{A}" width="21" height="19" />, <img src="README_images/f_B.png" align="center" border="0" alt=" f_{B}" width="21" height="19" />), they are trained on individual datasets <img src="README_images/S_A.png" align="center" border="0" alt="S_{A}" width="24" height="18" /> and <img src="README_images/S_B.png" align="center" border="0" alt="S_{B}" width="24" height="18" />.
  
**Mathematical definition for** <img src="README_images/sta_adv_r_est_func.png" align="center" border="0" alt="static\_adv\_rob\_estimation\big(S, f\big)" width="278" height="21" /> **function** <br />

  - <img src="README_images/est_func.png" align="center" border="0" alt="S_{B}" width="300" height="42" />
  
**Purpose of experiments** <br />

  - For each pair of machine learning models (<img src="README_images/f_A.png" align="center" border="0" alt=" f_{A}" width="21" height="19" />, <img src="README_images/f_B.png" align="center" border="0" alt=" f_{B}" width="21" height="19" />), we would like to experiment whether the relationship between <img src="README_images/sta_adv_r_A.png" align="center" border="0" alt="sta\_adv\_r_{A}" width="99" height="19" /> and <img src="README_images/sta_adv_r_B.png" align="center" border="0" alt="sta\_adv\_r_{B}" width="99" height="19" /> can indicate the relationship between actual adversarial robustness <img src="README_images/r_A.png" align="center" border="0" alt="r_{A}" width="21" height="15" /> and <img src="README_images/r_B.png" align="center" border="0" alt="r_{B}" width="21" height="15" /> (against state-of-art adversarial attacks). <br/> For instance, if <img src="README_images/sta_adv_r_A.png" align="center" border="0" alt="sta\_adv\_r_{A}" width="99" height="19" /> < <img src="README_images/sta_adv_r_B.png" align="center" border="0" alt="sta\_adv\_r_{B}" width="99" height="19" />, we expect to observe attack success rate on <img src="README_images/f_A.png" align="center" border="0" alt=" f_{A}" width="21" height="19" /> is higher than that of <img src="README_images/f_B.png" align="center" border="0" alt=" f_{B}" width="21" height="19" />.
  - Indicator function: <img src="README_images/indicator_func.png" align="center" border="0" alt="S_{B}" width="900" height="24" />
  
#### Multimedia classification task, MNIST

Experimental Settings | Match ratio | Time (sta_adv_r) | Time(r) | # of pairs | size(S) | Attack | Defense | eps
--- | --- | --- | --- |--- |--- |--- |--- |--- 
Trail 1 | 0.7000 | 0.00528 | 0.43242 | 100 | 600 | FGSM | None | 0.001  

Implementation Details:
1. Number of samples used for computing attack success rate: 1000
2. Architecture: Two layer (one hidden layer) ReLU (fully-connected) neural netowrk  
3. FGSM is generated according to the [Pytorch Official Website](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html). For more effective attacks, I will conduct those once I receive the accessibility of hardware resource. 
4. Actuacl robustness is calculated by 1 - attack sucess rate 
5. Both <img src="README_images/f_A.png" align="center" border="0" alt=" f_{A}" width="21" height="19" /> and <img src="README_images/f_B.png" align="center" border="0" alt=" f_{B}" width="21" height="19" /> are trained by the same architecture (include preprocess and activation functions). 

Next Steps: 
1. **(Running Experiment)** Measure on a set of architectures (Compare distribution instead of applying indicator function)
2. Use statistical approach (e.g., null test) instead of simple match ratio

Discussion: 
1. Design a dataset to have high sta_adv_r and another dataset to have low std_adv_r (Completed, but not actually meaningful) 
2. How to become more 'software-enginnering'?

Here we list out self-defined (related to our work) metrics that are correlated to (adversarial) robustness. 
1. Dimensionality [1]
2. Distance to decision boundary (in various directions like benign, adversarial, random)
3. Non-robust features [2]
4. Local intrinsic dimensionality [3]
5. Adversarial risk by the concentration of measure [4]