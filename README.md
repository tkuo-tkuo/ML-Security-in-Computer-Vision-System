# FYT
Final Year Thesis Project (COMP4981H) for Computer Science Students in HKUST


## Research Questions (RQs)

### RQ1: Can we design a metric to approximate the adversarial robustness efficiently?  

(Talk about the inspiration about relevant work) 
1. Dimensionality 
2. Distance to decision boundary (in various directions like benign, adversarial, random)
3. Non-robust features 
4. Some theoretical metrics. For instance, local intrinsic dimensionality and adversarial risk by the concentration of measure.

**Notations** <br />

  - The metric for adversarial robustness approximation is denoted as ***static estimation of adversarial risk***, <img src="README_images/sta_adv_r_est_formula.png" align="center" border="0" alt="sta\_adv\_r = static\_adv\_rob\_estimation\big(S, f\big) " width="369" height="21" />, where <img src="README_images/S.png" align="center" border="0" alt="S" width="17" height="15" /> and <img src="README_images/f.png" align="center" border="0" alt="f" width="12" height="19" /> indicate the training dataset and neural network trained. 
  - For each pair of machine learning models (<img src="README_images/f_A.png" align="center" border="0" alt=" f_{A}" width="21" height="19" />, <img src="README_images/f_B.png" align="center" border="0" alt=" f_{B}" width="21" height="19" />), they are trained on individual datasets <img src="README_images/S_A.png" align="center" border="0" alt="S_{A}" width="24" height="18" /> and <img src="README_images/S_B.png" align="center" border="0" alt="S_{B}" width="24" height="18" />.
  
**Purpose of experiments** <br />

  - For each pair of machine learning models (<img src="README_images/f_A.png" align="center" border="0" alt=" f_{A}" width="21" height="19" />, <img src="README_images/f_B.png" align="center" border="0" alt=" f_{B}" width="21" height="19" />), we would like to experiment whether the relationship between <img src="README_images/sta_adv_r_A.png" align="center" border="0" alt="sta\_adv\_r_{A}" width="99" height="19" /> and <img src="README_images/sta_adv_r_B.png" align="center" border="0" alt="sta\_adv\_r_{B}" width="99" height="19" /> can indicate the relationship between actual adversarial robustness <img src="README_images/r_A.png" align="center" border="0" alt="r_{A}" width="21" height="15" /> and <img src="README_images/r_B.png" align="center" border="0" alt="r_{B}" width="21" height="15" /> (against state-of-art adversarial attacks). <br/> For instance, if <img src="README_images/sta_adv_r_A.png" align="center" border="0" alt="sta\_adv\_r_{A}" width="99" height="19" /> < <img src="README_images/sta_adv_r_B.png" align="center" border="0" alt="sta\_adv\_r_{B}" width="99" height="19" />, we expect to observe attack success rate on <img src="README_images/f_A.png" align="center" border="0" alt=" f_{A}" width="21" height="19" /> is higher than that of <img src="README_images/f_B.png" align="center" border="0" alt=" f_{B}" width="21" height="19" />.
  - (Define indicator function for match ratio)
  
**Mathematical definition for** <img src="README_images/sta_adv_r_est_func.png" align="center" border="0" alt="static\_adv\_rob\_estimation\big(S, f\big)" width="278" height="21" /> **function** <br />

  - (Give mathmatical definition with appropriate notations) 


#### RQ1.1, (With training datasets extracted from different distribution)

(description for RQ1.1, about experimental settings for RQ1.1) <br />
(S_{A} and S_{B} should have distinct differences in terms of distribution in the input space, but the architecture should be the same for f_{A} and f_{B}) 

Experimental Settings | Match ratio | Time (sta_adv_r) | Time (sucess rate) | # of pairs | size(S) | S_A | S_B | Attack | Defense | eps
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |---
Trail 1 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 
Trail 2 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 
Trail 3 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 
Trail 4 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 
Trail 5 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 

(Specify number of samples used in computing attack success rate and architecture here)

#### RQ1.2 (With training datasets extracted from same distribution)  

(description for RQ1.2, about experimental settings for RQ1.2) <br />
(S_{A} and S_{B} should have be sampled from the same distribution in the input space. Also, architecture should remain the same for f_{A} and f_{B}) 

Experimental Settings | Match ratio | Time (sta_adv_r) | Time (sucess rate) | # of pairs | size(S) | S_A | S_B | Attack | Defense | eps
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |---
Trail 1 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 
Trail 2 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 
Trail 3 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 
Trail 4 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 
Trail 5 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 

## References 

some citations (or links)
