

    Experiment 5: Potential Method 1 & Integrated LPs judgement (adv_attack=i_FGSM, y/y'=y', model=CNN)
  
    Note that LP_i = B if risk_score_i < differentitation_line_i
    
    LP_1, LP_2, and LP_3 are LPs for the convolutional layers; LP_4 is the LP for the first ReLU layer. 

- If we intuitively set the differentiation lines and apply judgement rule (LP_1=A and LP_2=A) -> A, we can alreadly achieve 0% FPR and 13% FNR on CNN. 
- What if we see the distribution of risk scores so as to deliberately select differentiation lines and adv condition? <br/> Below figure represents the risk score distribution computed according to Potential Method 1. Even we only utilize LP_1 and set the differentiation line for LP_1 to be 300, it can differentiate all benign samples and most of adversarial samples. <br/> If we deliberately set the differentation lines to be [300, 320, 100, \_] and apply judgement rule (LP_1=B and LP_2=B and LP_3=B) -> B, we can achieve 9.2% FPR and 3.2% FNR.<br/>
<img src="../Images/Exp5/Exp5_1.png" align="center" border="0" width="414" height="554"/><br/>
- What if we compare each LP_i between benign and adversarial samples? Below figure demonstrates that for LP_1, LP_2, and LP_3, we can clearly differentiate benign samples and adversarial samples. However, by Potential Method 1, we are not capable of reaching 0% FPR and 0% FNR. <br/> Either FPR or FNR is 0%, then the other one will false error > 5%. <br/>
<img src="../Images/Exp5/exp5_2.png" align="center" border="0" width="864" height="576"/>
