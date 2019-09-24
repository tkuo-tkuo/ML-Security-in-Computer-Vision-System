# FYT
Final Year Thesis Project (COMP4981H) for Computer Science Students in HKUST


### Research Questions (RQs)

#### RQ1: As a developer, given the training dataset, can we design a metric to approximate the adversarial robustness efficiently?  

(Talk about the inspiration about relevant work) 
1. Dimensionality 
2. Distance to decision boundary (in various directions like benign, adversarial, random)
3. Non-robust features 
4. Some theoretical metrics. For instance, local intrinsic dimensionality and adversarial risk by the concentration of measure.

(Description the mathematical formula here)

Notations <br />
  - The metric for adversarial robustness approximation is denoted as ***static estimation of adversarial risk***, <img src="http://bit.ly/2mEjDTc" align="center" border="0" alt="sta\_adv\_r = static\_adv\_rob\_estimation\big(S, f\big) " width="386" height="21" />, where <img src="http://bit.ly/2lf1Phe" align="center" border="0" alt="(S, f)" width="44" height="19" /> indicates the training dataset and neural network trained. 
  - For each pair of machine learning models (<img src="http://www.sciweavers.org/tex2img.php?eq=%20f_%7BA%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt=" f_{A}" width="21" height="19" />, <img src="http://www.sciweavers.org/tex2img.php?eq=%20f_%7BB%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt=" f_{B}" width="21" height="19" />), they are trained on individual datasets <img src="http://bit.ly/2mgqpP2" align="center" border="0" alt="S_{A}" width="24" height="18" /> and <img src="http://bit.ly/2mh4M0W" align="center" border="0" alt="S_{B}" width="24" height="18" />.
  
Purpose of experiments <br />
  - For each pair of machine learning models (<img src="http://www.sciweavers.org/tex2img.php?eq=%20f_%7BA%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt=" f_{A}" width="21" height="19" />, we would liked to experiment whether the relationship between (sta_adv_r_{A} and sta_adv_r_{B}) indicates the relationship between actual adversarial robustness against state-of-art adversarial attacks. 

##### RQ1.1, (With training datasets extracted from different distribution)

(description for RQ1.1, about experimental settings for RQ1.1) <br />
(S_{A} and S_{B} should have distinct differences in terms of distribution in the input space, but the architecture should be the same for f_{A} and f_{B}) 

Experimental Settings | num of pairs | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Trail 1 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
Trail 2 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
Trail 3 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
Trail 4 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
Trail 5 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269

##### RQ1.2 (With training datasets extracted from same distribution)  

(description for RQ1.2, about experimental settings for RQ1.2) <br />
(S_{A} and S_{B} should have be sampled from the same distribution in the input space. Also, architecture should remain the same for f_{A} and f_{B}) 

Experimental Settings | num of pairs | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Trail 1 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
Trail 2 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
Trail 3 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
Trail 4 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
Trail 5 | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269

### References 

some citations (or links)
