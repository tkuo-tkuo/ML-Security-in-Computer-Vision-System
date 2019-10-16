# Final Year Thesis
Final Year Thesis Project (COMP4981H) for Computer Science Students in HKUST

## Research Questions (RQs)

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

#### _Approach 2: Leverage preconditions for the prediction postcondition in adversarial detection [5]_

#### Binary classification task (5 vs 7)

**Notations** <br />

- Training set: <img src="README_images/S.png" align="center" border="0" alt="S" width="17" height="15" />
- Subset (first class) of training set: S5
- Subset (second class) of training set: S7
- Precondition set: P
- Subset (first class) of precondition set: P5
- Subset (second class) of precondition set: P7

<details>
  <summary>Experiment 1: Detection via input preconditions on 1 hidden layer ReLU network</summary>

  ## 
  
  \|S\| | \|S5\| | \|S7\| | \|P5\| | \|P7\| | 1 - FPR | 1 - FNR | Input Augmentation
  --- | --- | --- | --- | --- | --- | --- | ---  
  500 | 227 | 273 | 71.090 (8.321) | 123.320 (14.118) | 63.8 (4.8)% | 32.3 (21.6)% | None
  3000 (500+2500) | 1362 | 1638 | 289.090 (24.717) | 644.110 (51.828) | 68.0 (3.5)% | 17.5 (12.0)% | Yes (Approach1)
  3000 (500+2500) | 1362 | 1638 | 365.170 (40.151) | 742.320 (93.065) | 68.3 (3.6)% | 14.8 (10.5)% | Yes (Approach2)
  1500 | 674 | 826 | 164.680 (28.098) | 223.410 (38.993) | 79.6 (4.2)% | 67.5 (17.4)% | None
  9000 (1500+7500) | 4044 | 4956 | 574.650 (82.479) | 1090.800 (186.220) | 83.7 (3.6)% | 52.4 (17.2)% | Yes (Approach1)
  9000 (1500+7500) | 4044 | 4956 | 682.920 (102.169) | 1378.340 (216.134) | 84.8 (3.4)% | 45.9 (15.8)% | Yes (Approach2)
  3000 | 1364 | 1636 | 425.950 (95.574) | 716.310 (145.208) | 69.6 (6.3)% | 97.9 (3)% | None
  18000 (3000+15000) | 8185 | 9815 | 1226.650 (331.550) | 3299.600 (682.530) | 74.5 (5.8)% | 92.7 (6.7)% | Yes (Approach1)
  18000 (3000+15000) | 8185 | 9815 | 1548.330 (359.290) | 3975.600 (833.274) | 74.7 (5.5)% | 89.2 (8.3)% | Yes (Approach2)
  
</details>

<details>
  <summary>Experiment 2: Detection via input preconditions on 2 hidden layer ReLU network</summary>

  ## 

  \|S\| | \|S5\| | \|S7\| | \|P5\| | \|P7\| | 1 - FPR | 1 - FNR | Input Augmentation
  --- | --- | --- | --- | --- | --- | --- | ---  
  500 | 227 | 273 | 102.920 (22.191) | 145.07 (27.156) | 57.9 (10.1)% | 46.8 (22.6)% | None
  3000 (500+2500) | 1362 | 1638 | 272.770 (63.876) | 469.120 (152.296) | 69.3 (7.3)% | 28.2 (16.0)% | Yes (Approach1)
  3000 (500+2500) | 1362 | 1638 | 338.890 (97.934) | 584.250 (181.626) | 68.0 (8.8)% | 28.2 (14.9)% | Yes (Approach2)
  1500 | 674 | 826 | 199.360 (62.468) | 279.480 (71.816) | 76.9 (7.5)% | 79.8 (18.7)% | None
  9000 (1500+7500) | 4044 | 4956 | 524.450 (161.528) | 914.390 (239.864) | 82.9 (5.3)% | 64.7 (22.9)% | Yes (Approach1)
  9000 (1500+7500) | 4044 | 4956 | 651.990 (205.734) | 1189.720 (363.669) | 82.9 (5.8)% | 58.7 (19.8)% | Yes (Approach2)
  3000 | 1364 | 1636 | 477.190 (121.253) | 683.120 (173.366) | 70.6 (6.6)% | 99.1 (2)% | None
  18000 (3000+15000) | 8185 | 9815 | 1205.820 (332.480) | 2549.280 (701.297) | 76.0 (6.3)% | 95.6 (6.9)% | Yes (Approach1)
  18000 (3000+15000) | 8185 | 9815 | 1427.990 (383.569) | 3360.290 (995.905) | 76.0 (7.4)% | 91.7 (8.3)% | Yes (Approach2)

</details>

<details>
  <summary>Experiment 3: Detection via input preconditions on 3 hidden layer ReLU network</summary>

  ## 

  \|S\| | \|S5\| | \|S7\| | \|P5\| | \|P7\| | 1 - FPR | 1 - FNR | Input Augmentation
  --- | --- | --- | --- | --- | --- | --- | ---  
  500 | 227 | 273 | 102.490 (22.727) | 126.32 (29.038) | 60.4 (8.6)% | 68.5 (21.1)% | None

</details>


Implementation Deatils:
1. All bengin and adversarial samples are generated according to MNIST dataset (size of 100)
2. Values in () indicate standard deviation 
3. Approach1 - append noise _~Uniform(lower_bound=-0.1, uppper_bound=0.1)_; 5 perturbed inputs are generated per input 
4. Approach2 - append noise _~Normal(mean=0, std=0.1)_; 5 perturbed inputs are generated per input 
5. Architecture: Two layer (one hidden layer) ReLU (fully-connected) neural netowrk  
6. Attack: iterative FGSM (attack until the perturbed input is misclassified)

Interesting Observations
<details>
  <summary>Observation 1: In the similar ReLU neural networks, we can observe the performance of detection increase dramatically with the increase of the complexity. To be more concise, when the model becomes more complicated, this approach can achieve lower FPR and FNR (by using provenance of the first layer).</summary>
    
  \|S\| | \|S5\| | \|S7\| | \|P5\| | \|P7\| | 1 - FPR | 1 - FNR | Input Augmentation | num of hidden layers
  --- | --- | --- | --- | --- | --- | --- | --- | ---
  500 | 227 | 273 | 71.090 (8.321) | 123.320 (14.118) | 63.8 (4.8)% | 32.3 (21.6)% | None | 1
  500 | 227 | 273 | 102.920 (22.191) | 145.07 (27.156) | 57.9 (10.1)% | 46.8 (22.6)% | None | 2
  500 | 227 | 273 | 102.490 (22.727) | 126.32 (29.038) | 60.4 (8.6)% | 68.5 (21.1)% | None | 3
  1500 | 674 | 826 | 164.680 (28.098) | 223.410 (38.993) | 79.6 (4.2)% | 67.5 (17.4)% | None | 1
  1500 | 674 | 826 | 199.360 (62.468) | 279.480 (71.816) | 76.9 (7.5)% | 79.8 (18.7)% | None | 2
  1500 | 674 | 826 | 204.580 (58.297) | 299.240 (96.728( | 73.7 (7.2)% | 94.0 (14.7)% | None | 3
  3000 | 1364 | 1636 | 425.950 (95.574) | 716.310 (145.208) | 69.6 (6.3)% | 97.9 (3)% | None | 1
  3000 | 1364 | 1636 | 477.190 (121.253) | 683.120 (173.366) | 70.6 (6.6)% | 99.1 (2)% | None | 2
  3000 | 1364 | 1636 | 502.220 (123.755) | 644.740 (170.400) | 70.7 (7.0)% | 99.96 (0.2%) | None | 3
  
</details>


**Objective** <br />

It should be able to achieve <b>low false negative (adversarial but regarded as benign) and acceptable false positive (benign but regarded as adversarial), which means that adversarial indentification rate should approach 100% and bengin indentification rate should acceptable.</b> 

## Appendix 

#### Appendix 1.1 
Here we list out self-defined (related to our work) metrics that are correlated to (adversarial) robustness. 

1. Dimensionality [1]
2. Distance to decision boundary (in various directions like benign, adversarial, random)
3. Non-robust features [2]
4. Local intrinsic dimensionality [3]
5. Adversarial risk by the concentration of measure [4]

## References 

[1] Florian Tramer, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick McDaniel. The space of transferable adversarial examples. arXiv preprint arXiv:1704.03453, 2017. <br />
[2] Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., and Madry, A. Adversarial examples are not bugs, they are features. arXiv preprint arXiv:1905.02175, 2019. <br />
[3] Ma, X., Li, B., Wang, Y., Erfani, S. M., Wijewickrema, S., Schoenebeck, G., Houle, M. E., Song, D., and Bailey, J. Characterizing adversarial subspaces using local intrinsic dimensionality. <br />
[4] Mahloujifar, S., Zhang, X., Mahmoody, M., and Evans, D. Empirically measuring concentration: Fundamental limits on intrinsic robustness. Safe Machine Learning workshop at ICLR, 2019. <br />
[5] Divya Gopinath, Hayes Converse, Corina S. Pasareanu, and Ankur Taly. Property Inference for Deep Neural Networks. ASE, 2019. <br />
[6] NIC. <br />
[7] Exploiting the Inherent Limitation of L0 Adversarial Examples <br />
