
    Experiment 6: Potential Method 2 & Integrated LPs judgement (adv_attack=i_FGSM, y/y'=y', model=CNN)

    Note that LP_i = B if risk_score_i < differentitation_line_i
    
    LP_1, LP_2, and LP_3 are LPs for the convolutional layers; LP_4 is the LP for the first ReLU layer. 
  
As shown in the following figures, it is difficult to tell that Potential Method 2 bring any improvement based to Potential Method 1. 

<div align="center">
LP_i risk score distribution with threshold=0.05 (i={1, 2, 3, 4}) 
<img src="../Images/Exp6/exp6_005.png" align="center" border="0" width="576" height="384"/>
</div>
<div align="center">
LP_i risk score distribution with threshold=0.1 (i={1, 2, 3, 4}) 
<img src="../Images/Exp6/exp6_01.png" align="center" border="0" width="576" height="384"/>
</div>
<div align="center">
LP_i risk score distribution with threshold=0.2 (i={1, 2, 3, 4}) 
<img src="../Images/Exp6/exp6_02.png" align="center" border="0" width="576" height="384"/>
</div>
<div align="center">
LP_i risk score distribution with threshold=0.3 (i={1, 2, 3, 4}) 
<img src="../Images/Exp6/exp6_03.png" align="center" border="0" width="576" height="384"/>
</div>
<div align="center">
LP_i risk score distribution with threshold=0.4 (i={1, 2, 3, 4}) 
<img src="../Images/Exp6/exp6_04.png" align="center" border="0" width="576" height="384"/>
</div>
<div align="center">
LP_i risk score distribution with threshold=0.5 (i={1, 2, 3, 4}) 
<img src="../Images/Exp6/exp6_05.png" align="center" border="0" width="576" height="384"/>
</div>
<div align="center">
LP_i risk score distribution with threshold=0.6 (i={1, 2, 3, 4}) 
<img src="../Images/Exp6/exp6_06.png" align="center" border="0" width="576" height="384"/>
</div>
<div align="center">
LP_i risk score distribution with threshold=0.7 (i={1, 2, 3, 4}) 
<img src="../Images/Exp6/exp6_07.png" align="center" border="0" width="576" height="384"/>
</div>
<div align="center">
LP_i risk score distribution with threshold=0.8 (i={1, 2, 3, 4}) 
<img src="../Images/Exp6/exp6_08.png" align="center" border="0" width="576" height="384"/>
</div>
<div align="center">
LP_i risk score distribution with threshold=0.9 (i={1, 2, 3, 4}) 
<img src="../Images/Exp6/exp6_09.png" align="center" border="0" width="576" height="384"/>
</div>