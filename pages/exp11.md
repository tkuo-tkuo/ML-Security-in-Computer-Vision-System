
    Experiment 11: Relation between PIs and dropout layer (adv_attack=i_FGSM, y/y'=y', model=CNN, qr=95, i-th_robustified_layer=1/3/4, approach=insertion and total retraining)</summary>
  
    Due to unsatisfied results of Exp 10, we would like to examinate what if we increase the dropout rate on other layers. 
    Below are diagrams to indicate influence of dropout layer before the first/third/fourth layer. 
    
    Before the first layer 
    
<img src="../Images/Exp11/1/exp11_1_1.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/1/exp11_2_1.png" width="650" height="140"/> 

<img src="../Images/Exp11/1/exp11_1_2.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/1/exp11_2_2.png" width="650" height="140"/>

<img src="../Images/Exp11/1/exp11_1_5.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/1/exp11_2_5.png" width="650" height="140"/>  

<img src="../Images/Exp11/1/exp11_1_10.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/1/exp11_2_10.png" width="650" height="140"/>

<img src="../Images/Exp11/1/exp11_1_20.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/1/exp11_2_20.png" width="650" height="140"/>

<img src="../Images/Exp11/1/exp11_1_30.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/1/exp11_2_30.png" width="650" height="140"/>

<img src="../Images/Exp11/1/exp11_1_40.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/1/exp11_2_40.png" width="650" height="140"/>

<img src="../Images/Exp11/1/exp11_1_50.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/1/exp11_2_50.png" width="650" height="140"/>
    
    Before the third layer 
    
<img src="../Images/Exp11/3/exp11_1_1.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/3/exp11_2_1.png" width="650" height="140"/> 

<img src="../Images/Exp11/3/exp11_1_2.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/3/exp11_2_2.png" width="650" height="140"/>

<img src="../Images/Exp11/3/exp11_1_5.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/3/exp11_2_5.png" width="650" height="140"/>  

<img src="../Images/Exp11/3/exp11_1_10.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/3/exp11_2_10.png" width="650" height="140"/>

<img src="../Images/Exp11/3/exp11_1_20.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/3/exp11_2_20.png" width="650" height="140"/>

<img src="../Images/Exp11/3/exp11_1_30.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/3/exp11_2_30.png" width="650" height="140"/>

<img src="../Images/Exp11/3/exp11_1_40.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/3/exp11_2_40.png" width="650" height="140"/>

<img src="../Images/Exp11/3/exp11_1_50.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/3/exp11_2_50.png" width="650" height="140"/>
    
    Before the fourth layer 
    
<img src="../Images/Exp11/4/exp11_1_1.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/4/exp11_2_1.png" width="650" height="140"/> 

<img src="../Images/Exp11/4/exp11_1_2.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/4/exp11_2_2.png" width="650" height="140"/>

<img src="../Images/Exp11/4/exp11_1_5.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/4/exp11_2_5.png" width="650" height="140"/>  

<img src="../Images/Exp11/4/exp11_1_10.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/4/exp11_2_10.png" width="650" height="140"/>

<img src="../Images/Exp11/4/exp11_1_20.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/4/exp11_2_20.png" width="650" height="140"/>

<img src="../Images/Exp11/4/exp11_1_30.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/4/exp11_2_30.png" width="650" height="140"/>

<img src="../Images/Exp11/4/exp11_1_40.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/4/exp11_2_40.png" width="650" height="140"/>

<img src="../Images/Exp11/4/exp11_1_50.png" align="left" border="0" width="150" height="140"/>
<img src="../Images/Exp11/4/exp11_2_50.png" width="650" height="140"/>

    Current Conclusion (2019 Dec. 3): increasing dropout will make benign and adversarial samples further indistinguishable!!
    It is also noticable that the weights of training and evaluating of models (which involve dropout layer) are different. 
    