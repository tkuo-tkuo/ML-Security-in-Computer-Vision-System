# Final Year Thesis
Final Year Thesis Project (COMP4981H) for Computer Science Students in HKUST

## Study on adversarial detection via LP (Layer Provenance)

### Notations

- **LP_i**: Layer Provenance of the i-th hidden layer
- **y**: ground-truth label, **y'** predicted label
- **S**: training set, **P**: provenance set
- **S1**: subset (first class) within training set, **S2**: subset (second class) within training set
- **P1**: subset (first class) within provenance set, **P2**: subset (second class) within provenance set
- **TPR**: True Positive Rate (A -> A)
- **TNR**: True Negative Rate (B -> B)
- **FPR**: False Positive Rate (B -> A)
- **FNR**: Flase Negative Rate (A -> B)
- **h**: Number of hidden layers (specifically for ReLU neural networks)
- **adv_a**: adversarial attack
- **i_FGSM**: Iterative Fast Gradient Sign Method, **JSMA**: Jacobian Saliency Map Attack, **CWL2**: CarliniWagner L2 Attack

### Expressions

- **()** indicate standard deviation. 

### Common Rules 

- All evaluations (TPR, TNR, FPR, and FNR) are examinated on 100 samples. 
- For Table 1 to 3 and Experiment 1 to 4, the task is to classify 5 and 7 (subset of MNIST). 
- For Table 1 to 3 and Experiment 1 to 4, if we use more than one LP, we will concatenate all LPs as one LP.  

### ReLU 

**Data collections (ReLU)** <br/>

<details>
  <summary>Table 1: TPR & TNR by LP_1 (adv_a=i_FGSM)</summary>
    
  \|S\| | \|S1\|/\|S2\| | \|P1\| | \|P2\| | TNR | TPR | h | y/y'
  --- | --- | --- | --- | --- | --- | --- | --- 
  500 | 227/273 | 70.850 (9.358) | 121.430 (15.163) | 64.0 (4.3)% | 34.5 (21.7)% | 1 | y
  500 | 227/273 | 99.480 (21.718) | 141.360 (29.135) | 59.1 (9.2)% | 43.2 (21.9)% | 2 | y
  500 | 227/273 | 103.550 (17.698) | 129.930 (26.887) | 59.9 (7.1)% | 70.0 (23.9)% | 3 | y
  500 | 227/273 | 96.940 (19.057) | 110.090 (29.264) | 65.1 (7.3)% | 72.9 (20.9)% | 4 | y
  1500 | 674/826 | 162.900 (24.819) | 223.570 (38.956) | 79.7 (4.3)% | 65.2 (16.1)% | 1 | y 
  1500 | 674/826 | 200.250 (52.630) | 262.800 (58.982) | 77.7 (5.8)% | 79.9 (19.8)% | 2 | y 
  1500 | 674/826 | 202.130 (59.237) | 301.580 (82.210) | 73.5 (6.7)% | 98.3 (3.1)% | 3 | y 
  1500 | 674/826 | 212.660 (57.575) | 279.020 (71.900) | 74.2 (6.6)% | 98.5 (3.8)% | 4 | y
  3000 | 1364/1636 | 432.980 (93.588) | 738.560 (175.844) | 68.8 (6.5)% | 98.2 (3)% | 1 | y 
  3000 | 1364/1636 | 463.520 (100.624) | 674.400 (170.379) | 71.4 (6.6)% | 99.1 (2.1)% | 2 | y
  3000 | 1364/1636 | 506.940 (127.957) | 674.990 (182.066) | 69.5 (7.2)% | 99.9 (0.5)% | 3 | y
  3000 | 1364/1636 | 490.720 (141.795) | 596.430 (180.541) | 71.9 (7.7)% | 99.9 (0.6)% | 4 | y
  500 | 227/273 | 70.480 (7.882) | 122.130 (14.576) | 64.1 (3.8)% | 18.6 (12.5)%| 1 | y'
  500 | 227/273 | 100.280 (20.691) | 145.170 (27.773) | 58.3 (9.7)% | 30.7 (18.5)% | 2 | y'
  500 | 227/273 | 106.030 (25.253) | 129.530 (28.993) | 59.2 (9.0)% | 55.4 (24.6)% | 3 | y'
  500 | 227/273 | 95.130 (21.880) | 108.630 (27.240) | 65.9 (8.1)% | 64.0 (25.0)% | 4 | y'
  1500 | 674/826 | 160.620 (27.222) | 223.630 (36.443) | 80.3 (3.6)% | 59.2 (18.2)% | 1 | y'
  1500 | 674/826 | 193.210 (56.364) | 285.100 (72.268) | 76.7 (7.4)% | 75.1 (20.8)% | 2 | y' 
  1500 | 674/826 | 209.590 (56.449) | 273.070 (77.071) | 74.3 (5.8)% | 95.9 (8.4)% | 3 | y'
  1500 | 674/826 | 199.280 (62.882) | 282.930 (73.903) | 74.5 (5.9)% | 96.0 (7.1)% | 4 | y' 
  3000 | 1364/1636 | 421.170 (102.090) | 755.510 (195.395) | 69.4 (7.2)% | 98.0 (3.9)% | 1 | y'
  3000 | 1364/1636 | 469.580 (127.705) | 698.100 (186.750) | 70.1 (7.5)% | 98.5 (3.0)% | 2 | y'
  3000 | 1364/1636 | 529.230 (137.620) | 662.250 (179.874) | 69.6 (6.6)% | 99.8 (0.4)% | 3 | y'
  3000 | 1364/1636 | 515.670 (144.309) | 604.660 (200.546) | 71.3 (7.3)% | 99.7 (0.7)% | 4 | y'
  
  
</details> 

<details>
  
  <summary>Table 2: TPR & TNR by LP_i combinations (adv_a=i_FGSM, h=4, y/y'=y)</summary>

  \|S\| | \|S1\|/\|S2\| | \|P1\| | \|P2\| | TNR | TPR | LP(s) | h
  --- | --- | --- | --- | --- | --- | --- | ---
  500 | 227/273 | 96.940 (19.057) | 110.090 (29.264) | 65.1 (7.3)% | 72.9 (20.9)% | 1 | 4
  500 | 227/273 | 99.160 (22.821) | 114.030 (29.648) | 64.0 (8.9)% | 75.8 (21.2)% | 1/2 | 4
  500 | 227/273 | 95.820 (25.080) | 108.020 (28.359) | 65.8 (9.0)% | 75.8 (21.4)% | 1/2/3 | 4
  500 | 227/273 | 99.030 (22.518) | 109.370 (28.731) | 64.6 (8.4)% | 74.3 (21.7)% | 1/2/3/4 | 4
  500 | 227/273 | 95.370 (23.317) | 111.200 (27.613) | 64.3 (8.6)% | 78.0 (19.9)% | 1/4 | 4
  500 | 227/273 | 14.300 (7.176) | 17.490 (7.640) | 96.1 (2.7)% | 30.1 (38.9)% | 2 | 4
  500 | 227/273 | 6.470 (2.364) | 5.890 (2.391) | 98.0 (2.2)% | 34.6 (37.2)% | 3 | 4
  500 | 227/273 | 3.890 (1.280) | 3.760 (1.320) | 97.8 (2.3)% | 46.6 (38.5)% | 4 | 4
  1500 | 674/826 | 212.660 (57.575) | 279.020 (71.900) | 74.2 (6.6)% | 98.5 (3.8)% | 1 | 4
  1500 | 674/826 | 208.290 (54.358) | 279.900 (84.229) | 74.3 (7.0)% | 98.6 (3.5)% | 1/2 | 4
  1500 | 674/826 | 205.810 (60.844) | 266.870 (80.710) | 74.8 (6.8)% | 98.9 (2.9)% | 1/2/3 | 4
  1500 | 674/826 | 210.150 (61.196) | 292.730 (91.176) | 73.5 (6.8)% | 99.0 (2.3)% | 1/2/3/4 | 4
  1500 | 674/826 | 208.620 (65.275) | 281.800 (80.535) | 74.2 (6.9)% | 98.4 (4.4)% | 1/4 | 4
  1500 | 674/826 | 19.280 (10.573) | 19.450 (9.583) | 97.9 (1.3)% | 76.0 (35.1)% | 2 | 4
  1500 | 674/826 | 6.460 (3.667) | 6.780 (3.657) | 99.5 (0.8)% | 56.3 (44.0)% | 3 | 4
  1500 | 674/826 | 3.260 (1.906) | 3.240 (1.550) | 99.5 (0.7)% | 74.2 (33.3)% | 4 | 4
  3000 | 1364/1636 | 490.720 (141.795) | 596.430 (180.541) | 71.9 (7.7)% | 99.9 (0.6)% | 1 | 4
  3000 | 1364/1636 | 514.680 (125.926) | 612.620 (189.286) | 70.5 (7.4)% | 100.0 (0.2)% | 1/2 | 4
  3000 | 1364/1636 | 521.110 (139.460) | 586.970 (179.797) | 71.1 (7.0)% | 100.0 (0.2)% | 1/2/3 | 4 
  3000 | 1364/1636 | 479.630 (127.896) | 590.500 (163.437) | 73.0 (6.9)% | 99.9 (0.3)% | 1/2/3/4 | 4
  3000 | 1364/1636 | 525.580 (161.507) | 617.430 (196.814) | 69.8 (8.2)% | 100.0 (0.3)% | 1/4 | 4 
  3000 | 1364/1636 | 25.500 (15.411) | 24.680 (14.115) | 98.8 (1.1)% | 81.8 (27.6)% | 2 | 4
  3000 | 1364/1636 | 5.510 (4.001) | 5.120 (3.179) | 99.8 (0.5)% | 88.8 (23.8)% | 3 | 4
  3000 | 1364/1636 | 1.770 (1.256) | 1.840 (1.111) | 99.9 (0.3)% | 95.7 (17.1)% | 4 | 4 

</details>

<details>
  <summary>Table 3: TPR & TNR by input augmentation (adv_a=i_FGSM, LPs=1, y/y'=y)</summary>
  
  **Notations** <br/>
  - **App_i**: Approach i
  - **Input_Aug**: Input Augmentation
  
  **Implementation details** <br/>
  - 5 perturbed inputs are generated per benign input
  - Input augmentation approach1 - append noise _~Uniform(lower_bound=-0.1, uppper_bound=0.1)_
  - Input augmentation approach2 - append noise _~Normal(mean=0, std=0.1)_

  \|S\| | \|S1\|/\|S2\| | \|P1\| | \|P2\| | TNR | TPR | Input_Aug | h
  --- | --- | --- | --- | --- | --- | --- | --- 
  500 | 227/273 | 70.850 (9.358) | 121.430 (15.163) | 64.0 (4.3)% | 34.5 (21.7)% | None | 1
  3000 (500+2500) | 1362/1638 | 289.090 (24.717) | 644.110 (51.828) | 68.0 (3.5)% | 17.5 (12.0)% | App_1 | 1
  3000 (500+2500) | 1362/1638 | 365.170 (40.151) | 742.320 (93.065) | 68.3 (3.6)% | 14.8 (10.5)% | App_2 | 1
  1500 | 674/826 | 162.900 (24.819) | 223.570 (38.956) | 79.7 (4.3)% | 65.2 (16.1)% | None | 1
  9000 (1500+7500) | 4044/4956 | 574.650 (82.479) | 1090.800 (186.220) | 83.7 (3.6)% | 52.4 (17.2)% | App_1 | 1
  9000 (1500+7500) | 4044/4956 | 682.920 (102.169) | 1378.340 (216.134) | 84.8 (3.4)% | 45.9 (15.8)% | App_2 | 1
  3000 | 1364/1636 | 432.980 (93.588) | 738.560 (175.844) | 68.8 (6.5)% | 98.2 (3)% | None | 1 
  18000 (3000+15000) | 8185/9815 | 1226.650 (331.550) | 3299.600 (682.530) | 74.5 (5.8)% | 92.7 (6.7)% | App_1 | 1
  18000 (3000+15000) | 8185/9815 | 1548.330 (359.290) | 3975.600 (833.274) | 74.7 (5.5)% | 89.2 (8.3)% | App_2 | 1
  500 | 227/273 | 99.480 (21.718) | 141.360 (29.135) | 59.1 (9.2)% | 43.2 (21.9)% | None | 2
  3000 (500+2500) | 1362/1638 | 272.770 (63.876) | 469.120 (152.296) | 69.3 (7.3)% | 28.2 (16.0)% | App_1 | 2
  3000 (500+2500) | 1362/1638 | 338.890 (97.934) | 584.250 (181.626) | 68.0 (8.8)% | 28.2 (14.9)% | App_2 | 2
  1500 | 674/826 | 200.250 (52.630) | 262.800 (58.982) | 77.7 (5.8)% | 79.9 (19.8)% | None | 2
  9000 (1500+7500) | 4044/4956 | 524.450 (161.528) | 914.390 (239.864) | 82.9 (5.3)% | 64.7 (22.9)% | App_1 | 2
  9000 (1500+7500) | 4044/4956 | 651.990 (205.734) | 1189.720 (363.669) | 82.9 (5.8)% | 58.7 (19.8)% | App_2 | 2
  3000 | 1364/1636 | 463.520 (100.624) | 674.400 (170.379) | 71.4 (6.6)% | 99.1 (2.1)% | None | 2
  18000 (3000+15000) | 8185/9815 | 1205.820 (332.480) | 2549.280 (701.297) | 76.0 (6.3)% | 95.6 (6.9)% | App_1 | 2
  18000 (3000+15000) | 8185/9815 | 1427.990 (383.569) | 3360.290 (995.905) | 76.0 (7.4)% | 91.7 (8.3)% | App_2 | 2
  
</details>

**Experiments  (ReLU)** 

<details>

  <summary>Experiment 1: Relationship between h and FPR & FNR (adv_a=i_FGSM, LPs=1, y/y'=y)</summary>
  
  <div align="center">
  FPR & FNR of adversarial detection with |S|=500 (h={1,2,3,4}) 
  </div>
  <img src="Images/Exp1/exp1_500.png" align="center" border="0" width="1200" height="170"/>
  <div align="center">
  FPR & FNR of adversarial detection with |S|=1500 (h={1,2,3,4}) 
  </div>
  <img src="Images/Exp1/exp1_1500.png" align="center" border="0" width="1200" height="170"/>
  <div align="center">
  FPR & FNR of adversarial detection with |S|=3000 (h={1,2,3,4}) 
  </div>
  <img src="Images/Exp1/exp1_3000.png" align="center" border="0" width="1200" height="170"/>
  
</details>

<details>
  
  <summary>Experiment 2: Relationship between |S| and FPR & FNR (adv_a=i_FGSM, LPs=1, y/y'=y)</summary>
  
  <div align="center">
  FPR & FNR of adversarial detection with h=1 (|S|={500,1500,3000}) 
  </div>
  <img src="Images/Exp2/exp2_1.png" align="center" border="0" width="1200" height="170"/>
  <div align="center">
  FPR & FNR of adversarial detection with h=2 (|S|={500,1500,3000}) 
  </div>
  <img src="Images/Exp2/exp2_2.png" align="center" border="0" width="1200" height="170"/>
  <div align="center">
  FPR & FNR of adversarial detection with h=3 (|S|={500,1500,3000}) 
  </div>
  <img src="Images/Exp2/exp2_3.png" align="center" border="0" width="1200" height="170"/>
  <div align="center">
  FPR & FNR of adversarial detection with h=4 (|S|={500,1500,3000}) 
  </div>
  <img src="Images/Exp2/exp2_4.png" align="center" border="0" width="1200" height="170"/>  
  
</details>

<details>
  
  <summary>Experiment 3: Relationship between single LP_i and FPR & FNR (adv_a=i_FGSM, y/y'=y)</summary>
  
  <div align="center">
  FPR & FNR of adversarial detection with |S|=500 (LP_i={LP_1,LP_2,LP_3,LP_4}) 
  </div>
  <img src="Images/Exp3/exp3_500.png" align="center" border="0" width="1200" height="170"/>
  <div align="center">
  FPR & FNR of adversarial detection with |S|=1500 (LP_i={LP_1,LP_2,LP_3,LP_4}) 
  </div>
  <img src="Images/Exp3/exp3_1500.png" align="center" border="0" width="1200" height="170"/>
  <div align="center">
  FPR & FNR of adversarial detection with |S|=3000 (LP_i={LP_1,LP_2,LP_3,LP_4}) 
  </div>
  <img src="Images/Exp3/exp3_3000.png" align="center" border="0" width="1200" height="170"/>
  
</details>

<details>
  
  <summary>Experiment 4: Relationship between LP_i combinations and FPR & FNR (adv_a=i_FGSM, y/y'=y)</summary>
  
  <div align="center">
  FPR & FNR of adversarial detection with |S|=500 (LP(s)={1,1/2,1/2/3,1/2/3/4}) 
  </div>
  <img src="Images/Exp4/exp4_500.png" align="center" border="0" width="1200" height="170"/>
  <div align="center">
  FPR & FNR of adversarial detection with |S|=1500 (LP(s)={1,1/2,1/2/3,1/2/3/4}) 
  </div>
  <img src="Images/Exp4/exp4_1500.png" align="center" border="0" width="1200" height="170"/>
  <div align="center">
  FPR & FNR of adversarial detection with |S|=3000 (LP(s)={1,1/2,1/2/3,1/2/3/4}) 
  </div>
  <img src="Images/Exp4/exp4_3000.png" align="center" border="0" width="1200" height="170"/>
  
</details>

**Observations  (ReLU)** <br/>
- Position of layers can influence detection capability. As we can see, when LP is closer to the end, TP  increases and TN decreases. One possible explanation is that when the LP is closer to the end, more samples (both for benign and adversarial samples) are likely to fall in the same provenance. 
- Different type of layers also have different detection capability. 
- We do not need to leverage all LPs. Single LP can achieve similar capability in terms of adversarial detection. 
- If LP_i is matched, LP_i+1 is extremely likely to be matched.
- An adversarial sample does not belong to either the provenance set of the ground-truth label or the provenance set of the predicted label
- y' class, both benign & adversarial samples on 4 hidden layers ReLU → [A, B, B, B] or [A, A, B, B]
- y class, most then [B, B, B, B] or [A, A, A, A]

### CNN

**Data collections (CNN)** 

(to be updated) 

**Experiments (CNN)** 

<details>
  <summary>Experiment 5: Potential Method 1 & Integrated LPs judgement (adv_attack=i_FGSM, y/y'=y', model=CNN)</summary>
<br/>
  
    Note that LP_i = B if risk_score_i < differentitation_line_i
    
    LP_1, LP_2, and LP_3 are LPs for the convolutional layers; LP_4 is the LP for the first ReLU layer. 
  
  - If we intuitively set the differentiation lines and apply judgement rule (LP_1=A and LP_2=A) -> A, we can alreadly achieve 0% FPR and 13% FNR on CNN. 
  - What if we see the distribution of risk scores so as to deliberately select differentiation lines and adv condition? <br/> Below figure represents the risk score distribution computed according to Potential Method 1. Even we only utilize LP_1 and set the differentiation line for LP_1 to be 300, it can differentiate all benign samples and most of adversarial samples. <br/> If we deliberately set the differentation lines to be [300, 320, 100, \_] and apply judgement rule (LP_1=B and LP_2=B and LP_3=B) -> B, we can achieve 9.2% FPR and 3.2% FNR.<br/>
  <img src="Images/Exp5/Exp5_1.png" align="center" border="0" width="414" height="554"/><br/>
  - What if we compare each LP_i between benign and adversarial samples? Below figure demonstrates that for LP_1, LP_2, and LP_3, we can clearly differentiate benign samples and adversarial samples. However, by Potential Method 1, we are not capable of reaching 0% FPR and 0% FNR. <br/> Either FPR or FNR is 0%, then the other one will false error > 5%. <br/>
  <img src="Images/Exp5/exp5_2.png" align="center" border="0" width="864" height="576"/>

</details>

<details>
  <summary>Experiment 6: Potential Method 2 & Integrated LPs judgement (adv_attack=i_FGSM, y/y'=y', model=CNN)</summary>
<br/>
  
    Note that LP_i = B if risk_score_i < differentitation_line_i
    
    LP_1, LP_2, and LP_3 are LPs for the convolutional layers; LP_4 is the LP for the first ReLU layer. 
  
  As shown in the following figures, it is difficult to tell that Potential Method 2 bring any improvement based to Potential Method 1. 
  
  <div align="center">
  LP_i risk score distribution with threshold=0.05 (i={1, 2, 3, 4}) 
  <img src="Images/Exp6/exp6_005.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with threshold=0.1 (i={1, 2, 3, 4}) 
  <img src="Images/Exp6/exp6_01.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with threshold=0.2 (i={1, 2, 3, 4}) 
  <img src="Images/Exp6/exp6_02.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with threshold=0.3 (i={1, 2, 3, 4}) 
  <img src="Images/Exp6/exp6_03.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with threshold=0.4 (i={1, 2, 3, 4}) 
  <img src="Images/Exp6/exp6_04.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with threshold=0.5 (i={1, 2, 3, 4}) 
  <img src="Images/Exp6/exp6_05.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with threshold=0.6 (i={1, 2, 3, 4}) 
  <img src="Images/Exp6/exp6_06.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with threshold=0.7 (i={1, 2, 3, 4}) 
  <img src="Images/Exp6/exp6_07.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with threshold=0.8 (i={1, 2, 3, 4}) 
  <img src="Images/Exp6/exp6_08.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with threshold=0.9 (i={1, 2, 3, 4}) 
  <img src="Images/Exp6/exp6_09.png" align="center" border="0" width="576" height="384"/>
  </div>
 
</details>

<details>
  <summary>Experiment 7: Potential Method 3 & Integrated LPs judgement (adv_attack=i_FGSM, y/y'=y', model=CNN)</summary>
<br/>
  
    Note that LP_i = B if B_log_prob_i > log_prob_diff_line_i
    
    LP_1, LP_2, and LP_3 are LPs for the convolutional layers; LP_4 is the LP for the first ReLU layer. 
    
  As shown in the following figures, we can observe that Potential Method 3 also achieve the same functionality to separate benign and adversarial samples as Potential Method 1. However, similar as Potential Method 1, we still not yet achieve 0% FPR and 0% FNR. 

  <div align="center">
  LP_i risk score distribution with |S|=1000 (i={1, 2, 3, 4}) 
  <img src="Images/Exp7/exp7_1000.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with |S|=5000 (i={1, 2, 3, 4}) 
  <img src="Images/Exp7/exp7_5000.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with |S|=10000 (i={1, 2, 3, 4}) 
  <img src="Images/Exp7/exp7_10000.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with |S|=15000 (i={1, 2, 3, 4}) 
  <img src="Images/Exp7/exp7_15000.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with |S|=20000 (i={1, 2, 3, 4}) 
  <img src="Images/Exp7/exp7_20000.png" align="center" border="0" width="576" height="384"/>
  </div>
  
</details>

<details>
  <summary>Experiment 8: Potential Method 4 & Integrated LPs judgement (adv_attack=i_FGSM, y/y'=y', model=CNN)</summary>
<br/>
  
    Note that LP_i = B if B_log_prob_i > log_prob_diff_line_i
    
    LP_1, LP_2, and LP_3 are LPs for the convolutional layers; LP_4 is the LP for the first ReLU layer. 
  
  As shown in the following figures, it is difficult to tell that Potential Method 4 bring any improvement based to Potential Method 3. 
  
  <div align="center">
  LP_i risk score distribution with delta=0.1 (i={1, 2, 3, 4}) 
  <img src="Images/Exp8/exp8_01.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with delta=0.2 (i={1, 2, 3, 4}) 
  <img src="Images/Exp8/exp8_02.png" align="center" border="0" width="576" height="384"/>
  </div>  
  <div align="center">
  LP_i risk score distribution with delta=0.3 (i={1, 2, 3, 4}) 
  <img src="Images/Exp8/exp8_03.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with delta=0.4 (i={1, 2, 3, 4}) 
  <img src="Images/Exp8/exp8_04.png" align="center" border="0" width="576" height="384"/>
  </div>
  <div align="center">
  LP_i risk score distribution with delta=0.45 (i={1, 2, 3, 4}) 
  <img src="Images/Exp8/exp8_045.png" align="center" border="0" width="576" height="384"/>
  </div>
  
</details>


**Observations (CNN)** <br/>
- (to be updated)

### To-Do List 

High Priority
- <b>~~LPs separation (no concatenation)~~</b>
- <b>~~Change the implementation for 10 classes (MNIST)~~</b> 
- <b>~~Experiment on CNN models~~</b>
- <b>(Wokring) Investigate whether most of adversarial samples fall into the same provenance. If it is the case, maybe we can remove the provenance where adversarial samples are potential to match -> draw a graph to demonstrate the result</b>
- Experiment on widely-adopted models 
- Store part of trained models as 'h5' format to preserve reproducibility

Low Priority
- Unify the format of citations
- Enlarge the table to include 0qr(min)/1qr/2qr(medium)/3qr/4qr(max)
- Document down the model architectures 

## Appendix 

#### Appendix 1.1 Architectures 

<details>
  <summary>Architectures of models (all ReLU networks)</summary>
  
  Jotting for architectures (More specification illustration required)
  - 784 64 2 (1)
  - 784 64 10 2 (2)
  - 784 64 32 10 2 (3)
  - 784 64 32 20 10 2 (4)
  
</details>

<details>
  <summary>Architectures of models (all CNN networks)</summary>
  
  (more)
  
</details>

<details>
  <summary>List of widely-adopted networks examinated</summary>
  
  (more)
  
</details>

#### Appendix 1.2 Original Rule-based Method & 4 potential improvements 

Note that all mathematical formulas are written for <b>only a layer</b> in a given neural network instead of the whole neural network.  

<details>
  
  <summary>Original Rule-based Method: Determine whether the provenance of input is learned</summary>
  
  <img src="README_images/original_method.png" align="center" border="0" width="900" height="121"/>

</details>

<details>
  
  <summary>Potential Method 1: Compute SP (Sample Probability) for each neuron & Determine whether l1-distance between SP(LP) and provenance of input is close enough</summary>
  
  <img src="README_images/potential_method_1.png" align="center" border="0" width="900" height="121"/>

</details>

<details>
  
  <summary>Potential Method 2: Compute SP (Sample Probability) for each neuron & Determine whether l1-distance between SP(LP) and provenance of input is close enough, where we filter out neurons that is relatively close (< beta).</summary>
  
  <img src="README_images/potential_method_2.png" align="center" border="0" width="900" height="226"/>

</details>

<details>
  
  <summary>Potential Method 3: Compute the probability that each neuron to be benign & Multiple all probabilities of neurons to determine whether a given input is benign/adversarial by a probability threshold value.</summary>

  <img src="README_images/potential_method_3.png" align="center" border="0" width="900" height="272"/>
  
</details>

<details>
  
  <summary>Potential Method 4: Compute the probability that each neuron to be benign & Multiple all probabilities of neurons to determine whether a given input is benign/adversarial by a probability threshold value, where we filter out neurons with probabilities relatively ambigious (e.g., 0.3 - 0.7)</summary>

  <img src="README_images/potential_method_4.png" align="center" border="0" width="900" height="343"/>

</details>


#### Appendix 1.3 Interesting Trials 

Use **probability absolute difference method** instead of rule-based (belonging judgement) method. 

\|S\| | \|S1\|/\|S2\| | \|P1\| | \|P2\| | TNR | TPR | LP(s) | h | alpha 
--- | --- | --- | --- | --- | --- | --- | --- | ---
3000 | 1364/1636 | 495.060 (128.945) | 616.210 (163.089) | 83.4 (10.4)% | 96.6 (5.7)% | 1 | 4 | 10


## References 
[1] Florian Tramer, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick McDaniel. The space of transferable adversarial examples. arXiv preprint arXiv:1704.03453, 2017. <br />
[2] Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., and Madry, A. Adversarial examples are not bugs, they are features. arXiv preprint arXiv:1905.02175, 2019. <br />
[3] Ma, X., Li, B., Wang, Y., Erfani, S. M., Wijewickrema, S., Schoenebeck, G., Houle, M. E., Song, D., and Bailey, J. Characterizing adversarial subspaces using local intrinsic dimensionality. <br />
[4] Mahloujifar, S., Zhang, X., Mahmoody, M., and Evans, D. Empirically measuring concentration: Fundamental limits on intrinsic robustness. Safe Machine Learning workshop at ICLR, 2019. <br />
[5] Divya Gopinath, Hayes Converse, Corina S. Pasareanu, and Ankur Taly. Property Inference for Deep Neural Networks. ASE, 2019. <br />
[6] NIC. <br />
[7] Exploiting the Inherent Limitation of L0 Adversarial Examples <br />
