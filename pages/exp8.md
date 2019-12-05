
    Experiment 8: Potential Method 4 & Integrated LPs judgement (adv_attack=i_FGSM, y/y'=y', model=CNN)
  
    Note that LP_i = B if B_log_prob_i > log_prob_diff_line_i
    
    LP_1, LP_2, and LP_3 are LPs for the convolutional layers; LP_4 is the LP for the first ReLU layer. 

As shown in the following figures, it is difficult to tell that Potential Method 4 bring any improvement based to Potential Method 3. 

<div align="center">
LP_i risk score distribution with delta=0.1 (i={1, 2, 3, 4}) 
<img src="../Images/Exp8/exp8_01.png" align="center" border="0" width="768" height="512"/>
</div>
<div align="center">
LP_i risk score distribution with delta=0.2 (i={1, 2, 3, 4}) 
<img src="../Images/Exp8/exp8_02.png" align="center" border="0" width="768" height="512"/>
</div>  
<div align="center">
LP_i risk score distribution with delta=0.3 (i={1, 2, 3, 4}) 
<img src="../Images/Exp8/exp8_03.png" align="center" border="0" width="768" height="512"/>
</div>
<div align="center">
LP_i risk score distribution with delta=0.4 (i={1, 2, 3, 4}) 
<img src="../Images/Exp8/exp8_04.png" align="center" border="0" width="768" height="512"/>
</div>
<div align="center">
LP_i risk score distribution with delta=0.45 (i={1, 2, 3, 4}) 
<img src="../Images/Exp8/exp8_045.png" align="center" border="0" width="768" height="512"/>
</div>
