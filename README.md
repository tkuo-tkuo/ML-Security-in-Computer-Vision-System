# FYT
Final Year Thesis Project (COMP4981H) for Computer Science Students in HKUST


## Research Questions (RQs)

### RQ1: Can we design a metric to approximate the adversarial robustness efficiently?  

#### Approach 1: Directly estimate the training dataset

There are related papers, which aim to define (theoretical) metrix that highly correlated to robustness. I place description of relevant work in Appendix 1.1. 

**Notations** <br />

  - The metric for adversarial robustness approximation is denoted as ***static estimation of adversarial risk***, <img src="README_images/sta_adv_r_est_formula.png" align="center" border="0" alt="sta\_adv\_r = static\_adv\_rob\_estimation\big(S, f\big) " width="369" height="21" />, where <img src="README_images/S.png" align="center" border="0" alt="S" width="17" height="15" /> and <img src="README_images/f.png" align="center" border="0" alt="f" width="12" height="19" /> indicate the training dataset and neural network trained. 
  - For each pair of machine learning models (<img src="README_images/f_A.png" align="center" border="0" alt=" f_{A}" width="21" height="19" />, <img src="README_images/f_B.png" align="center" border="0" alt=" f_{B}" width="21" height="19" />), they are trained on individual datasets <img src="README_images/S_A.png" align="center" border="0" alt="S_{A}" width="24" height="18" /> and <img src="README_images/S_B.png" align="center" border="0" alt="S_{B}" width="24" height="18" />.
  
**Mathematical definition for** <img src="README_images/sta_adv_r_est_func.png" align="center" border="0" alt="static\_adv\_rob\_estimation\big(S, f\big)" width="278" height="21" /> **function** <br />

  - (Give mathmatical definition with appropriate notations) 
  
**Purpose of experiments** <br />

  - For each pair of machine learning models (<img src="README_images/f_A.png" align="center" border="0" alt=" f_{A}" width="21" height="19" />, <img src="README_images/f_B.png" align="center" border="0" alt=" f_{B}" width="21" height="19" />), we would like to experiment whether the relationship between <img src="README_images/sta_adv_r_A.png" align="center" border="0" alt="sta\_adv\_r_{A}" width="99" height="19" /> and <img src="README_images/sta_adv_r_B.png" align="center" border="0" alt="sta\_adv\_r_{B}" width="99" height="19" /> can indicate the relationship between actual adversarial robustness <img src="README_images/r_A.png" align="center" border="0" alt="r_{A}" width="21" height="15" /> and <img src="README_images/r_B.png" align="center" border="0" alt="r_{B}" width="21" height="15" /> (against state-of-art adversarial attacks). <br/> For instance, if <img src="README_images/sta_adv_r_A.png" align="center" border="0" alt="sta\_adv\_r_{A}" width="99" height="19" /> < <img src="README_images/sta_adv_r_B.png" align="center" border="0" alt="sta\_adv\_r_{B}" width="99" height="19" />, we expect to observe attack success rate on <img src="README_images/f_A.png" align="center" border="0" alt=" f_{A}" width="21" height="19" /> is higher than that of <img src="README_images/f_B.png" align="center" border="0" alt=" f_{B}" width="21" height="19" />.
  - (Define indicator function for match ratio)

#### Multimedia classification task, MNIST

Specify 
1. number of samples used in computing attack success rate
2. architecture
3. distribution of S_{A} and S_{B} 

Experimental Settings | Match ratio | Time (sta_adv_r) | Time(r) | # of pairs | size(S) | Attack | Defense | eps
--- | --- | --- | --- |--- |--- |--- |--- |--- 
Trail 1 | 0.7000 | 0.00528 | 0.43242 | 100 | 500 | FGSM | None | 0.001  
Trail 2 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 
Trail 3 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 

(Describle how eps is related to the FGSM -> refer to the official Pytorch website) <br />
(Describle that the unit of Time is second) <br />
(Describle that r is actually 1 - attack sucess rate) <br />
(Descirble that attack success rate is based on adversarial attacks genereated by test dataset, where we randomly select 1000 samples.) <br />
(Descirble that you should ensure both f_{A} and f_{B} have similar capability, which is simply accurancy. In our experiments, one feasible approach is to train models approachine 100% accurancy) <br />
(Same test dataset should be applied to verify f_{A} and f_{B})

#### Approach 2: Leverage preconditions for the prediction postcondition in adversarial detection [5]

#### Binary classification task (5 vs 7)

num of train set | num of test set | num of samples for 5 | num of samples for 7 | num of precondition set for 5 | num of precondition set for 7 | Benign Indentification Rate | Adversarial Indentification Rate | Is Purely Noised Inputs Included?
--- | --- | --- | --- |--- |--- |--- |--- |--- 
500 | 100 | 227 | 273 | 74 | 105 | 65% | 31% | None
3000 (500+2500) | 100 | 1368 | 1632 | 485 | 129 | 65% | 43% | Yes (5 per original input)
1500 | 100 | 674 | 826 | 168 | 200 | 82% | 52% | None
9000 (1500+7500) | 100 | 4393 | 4607 | 890 | 225 | 85% | 23% | Yes (5 per original input)
3000 | 100 | 1364 | 1636 | 374 | 643 | 79% | 100% | None
18000 (3000+15000) | 100 | 9970 | 8030 | 1539 | 750 | 82% | 81% | Yes (5 per original input)
7500 | 100 | 3495 | 4005 | 1690 | 2037 | 63% | 100% | None
45000 (7500+37500) | 100 | 31159 | 13841 | 7173 | 3618 | 63% | 97% | Yes (5 per original input)

(Currently, noise is added by uniform distribution & perturbed images are based on images from test dataset)
(Should try: normal distribution & FGSM generated from pure random images)

(Talk about the attack: iterative FGSM instead of fix-epi FGSM attack) <br />
(Talk about the model used) <br />
(Talk about what is purely noised inputs) <br />
(Talk about False positive (judge benign as adversarial) and False negative (judge adversarial as benign)) <br />
(Talk about that actucally False negative is way more important than False positive) <br />

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
[5] Divya Gopinath, Hayes Converse, Corina S. Pasareanu, and Ankur Taly. Property Inference for Deep Neural Networks. ASE, 2019. 
