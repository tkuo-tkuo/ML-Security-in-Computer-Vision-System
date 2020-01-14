# Final Year Thesis
Final Year Thesis Project (COMP4981H) for Computer Science Students in HKUST

## Research on adversarial detection/robustification via LP (Layer Provenance)

<details>
  <summary>Notations & Expressions</summary>

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
- **()** indicate standard deviation. 

</details>

<details>
  <summary>Common rules</summary>

- For Table 1 to 3 and Experiment 1 to 4TPR, TNR, FPR, and FNR are examinated on 100 samples. 
- For Table 1 to 3 and Experiment 1 to 4, the task is to classify 5 and 7 (subset of MNIST). 
- For Table 1 to 3 and Experiment 1 to 4, if we use more than one LP, we will concatenate all LPs as one LP.  

</details>

### ReLU 

**Experimental data collections (ReLU)** <br/>

[Table 1: TPR & TNR by LP_1 (adv_a=i_FGSM)](pages/table1.md)

[Table 2: TPR & TNR by LP_i combinations (adv_a=i_FGSM, h=4, y/y'=y)](pages/table2.md)

[Table 3: TPR & TNR by input augmentation (adv_a=i_FGSM, LPs=1, y/y'=y)](pages/table3.md)

**Experiments  (ReLU)** 

[Exp 1: Relationship between h and FPR & FNR (adv_a=i_FGSM, LPs=1, y/y'=y)](pages/exp1.md)

[Exp 2: Relationship between |S| and FPR & FNR (adv_a=i_FGSM, LPs=1, y/y'=y)](pages/exp2.md)

[Exp 3: Relationship between single LP_i and FPR & FNR (adv_a=i_FGSM, y/y'=y)](pages/exp3.md)

[Exp 4: Relationship between LP_i combinations and FPR & FNR (adv_a=i_FGSM, y/y'=y)](pages/exp4.md)

<details>
  <summary>Observations (ReLU)</summary>

  - Position of layers can influence detection capability. As we can see, when LP is closer to the end, TP  increases and TN decreases. One possible explanation is that when the LP is closer to the end, more samples (both for benign and adversarial samples) are likely to fall in the same provenance. 
  - Different type of layers also have different detection capability. 
  - We do not need to leverage all LPs. Single LP can achieve similar capability in terms of adversarial detection. 
  - If LP_i is matched, LP_i+1 is extremely likely to be matched.
  - An adversarial sample does not belong to either the provenance set of the ground-truth label or the provenance set of the predicted label
  - y' class, both benign & adversarial samples on 4 hidden layers ReLU → [A, B, B, B] or [A, A, B, B]
  - y class, most then [B, B, B, B] or [A, A, A, A]

</details>

### CNN

**Experiments (CNN)**

[Exp 5: Potential Method 1 & Integrated LPs judgement (adv_attack=i_FGSM, y/y'=y', model=CNN)](pages/exp5.md)

[Exp 6: Potential Method 2 & Integrated LPs judgement (adv_attack=i_FGSM, y/y'=y', model=CNN)](pages/exp6.md)

[Exp 7: Potential Method 3 & Integrated LPs judgement (adv_attack=i_FGSM, y/y'=y', model=CNN)](pages/exp7.md)

[Exp 8: Potential Method 4 & Integrated LPs judgement (adv_attack=i_FGSM, y/y'=y', model=CNN)](pages/exp8.md)

[Exp 9: Relation between percentile (PCTL/qr) differentation line and 'Classified Benign Ratio' (CBR) (adv_attack=i_FGSM, y/y'=y', model=CNN)](pages/exp9.md)

[Exp10: Relation between PIs and dropout layer (adv_attack=i_FGSM, y/y'=y', model=CNN, qr=95, i-th_robustified_layer=2, approach=insertion and total retraining)](pages/exp10.md)

[Exp11: Relation between PIs and dropout layer (adv_attack=i_FGSM, y/y'=y', model=CNN, qr=95, i-th_robustified_layer=1/3/4, approach=insertion and total retraining)](pages/exp11.md)

[Exp12: Relation between PIs and dropout layer (adv_attack=i_FGSM, y/y'=y', model=CNN, qr=95, approach=insertion only)](pages/exp12.md)

[Exp13: Relation between PIs and dropout layer (adv_attack=i_FGSM, y/y'=y', model=CNN, qr=95, approach=insertion and slight tuning)](pages/exp13.md)

[Exp14: (Enhanced) Relation between PIs and dropout rate (before last layer), 0.05 interval to 0.7  (adv_attack=i_FGSM, y/y'=y', model=CNN, qr=95, approach=insertion and slight tuning)](pages/exp14.md) 

[Exp15: (Enhanced) Relation between PIs and dropout rate (before last layer), 0.01 interval to 0.2  (adv_attack=i_FGSM, y/y'=y', model=CNN, qr=95, approach=insertion and slight tuning)](pages/exp15.md) 

[Exp16: Relation between dropout rate (before last year) and ASR (adv_attack=i_FGSM, y/y'=y', model=CNN, qr=95, approach=insertion and slight tuning)](pages/exp16.md) 

**Observations (CNN)** <br/>
- (to be updated)

### To-Do List 

[To-Do list page](pages/todo.md)

## Appendix 

[Appendix 1.1 Architectures](pages/appendix1_1.md) 

[Appendix 1.2 Original Rule-based Method & 4 potential improvements](pages/appendix1_2.md)

## References 
[1] Florian Tramer, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick McDaniel. The space of transferable adversarial examples. arXiv preprint arXiv:1704.03453, 2017. <br />
[2] Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., and Madry, A. Adversarial examples are not bugs, they are features. arXiv preprint arXiv:1905.02175, 2019. <br />
[3] Ma, X., Li, B., Wang, Y., Erfani, S. M., Wijewickrema, S., Schoenebeck, G., Houle, M. E., Song, D., and Bailey, J. Characterizing adversarial subspaces using local intrinsic dimensionality. <br />
[4] Mahloujifar, S., Zhang, X., Mahmoody, M., and Evans, D. Empirically measuring concentration: Fundamental limits on intrinsic robustness. Safe Machine Learning workshop at ICLR, 2019. <br />
[5] Divya Gopinath, Hayes Converse, Corina S. Pasareanu, and Ankur Taly. Property Inference for Deep Neural Networks. ASE, 2019. <br />
[6] Shiqing Ma, Yingqi Liu, Guanhong Tao, Wen-Chuan Lee, and Xiangyu Zhang. 2019. “NIC: Detecting Adversarial Samples with Neural Network Invariant Checking” in Proceedings of the 26th Network and Distributed System Security Symposium, 2019. <br />

