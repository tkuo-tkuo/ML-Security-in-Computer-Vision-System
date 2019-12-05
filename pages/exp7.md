
    Experiment 7: Potential Method 3 & Integrated LPs judgement (adv_attack=i_FGSM, y/y'=y', model=CNN)
  
    Note that LP_i = B if B_log_prob_i > log_prob_diff_line_i
    
    LP_1, LP_2, and LP_3 are LPs for the convolutional layers; LP_4 is the LP for the first ReLU layer. 
    
As shown in the following figures, we can observe that Potential Method 3 also achieve the same functionality to separate benign and adversarial samples as Potential Method 1. However, similar as Potential Method 1, we still not yet achieve 0% FPR and 0% FNR. 

<div align="center">
LP_i risk score distribution with |S|=1000 (i={1, 2, 3, 4}) 
<img src="../Images/Exp7/exp7_1000.png" align="center" border="0" width="768" height="512"/>
</div>
<div align="center">
LP_i risk score distribution with |S|=5000 (i={1, 2, 3, 4}) 
<img src="../Images/Exp7/exp7_5000.png" align="center" border="0" width="768" height="512"/>
</div>
<div align="center">
LP_i risk score distribution with |S|=10000 (i={1, 2, 3, 4}) 
<img src="../Images/Exp7/exp7_10000.png" align="center" border="0" width="768" height="512"/>
</div>
<div align="center">
LP_i risk score distribution with |S|=15000 (i={1, 2, 3, 4}) 
<img src="../Images/Exp7/exp7_15000.png" align="center" border="0" width="768" height="512"/>
</div>
<div align="center">
LP_i risk score distribution with |S|=20000 (i={1, 2, 3, 4}) 
<img src="../Images/Exp7/exp7_20000.png" align="center" border="0" width="768" height="512"/>
</div>