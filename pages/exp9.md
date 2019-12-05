    Experiment 9: Relation between percentile (PCTL/qr) differentation line and 'Classified Benign Ratio' (CBR) (adv_attack=i_FGSM, y/y'=y', model=CNN)
    
  <br/>
  <div>
  CBR in LP_i layer with <b>qr=0.90</b> (i={1, 2, 3, 4}) <br/>
    
  Input | CBR_L1 | CBR_L2 | CBR_L3 | CBR_L4 
  --- | --- | --- | --- | --- 
  Train dataset (benign) | 0.94949495 | 0.90909091 | 0.93939394 | 0.93939394
  Test dataset (benign) | 0.90816327 | 0.93877551 | 0.90816327 | 0.89795918
  Test dataset (adv) | 0.10638298 | 0.39361702 | 0.06382979 | 0.9893617
    
  <img src="../Images/Exp9/exp9_90.png" align="center" border="0" width="576" height="384"/>
  </div>
  <br/>
  <div>
  CBR in LP_i layer with <b>qr=0.95</b> (i={1, 2, 3, 4}) <br/>
    
  Input | CBR_L1 | CBR_L2 | CBR_L3 | CBR_L4 
  --- | --- | --- | --- | --- 
  Train dataset (benign) | 0.96969697 | 0.94949495 | 0.98989899 | 0.94949495
  Test dataset (benign) | 0.96938776 | 0.95918367 | 0.94897959 | 0.95918367
  Test dataset (adv) | 0.10638298 | 0.5106383 | 0.17021277 | 0.9893617
    
  <img src="../Images/Exp9/exp9_95.png" align="center" border="0" width="576" height="384"/>
  </div>
  <br/>
  <div>
  CBR in LP_i layer with <b>qr=0.96</b> (i={1, 2, 3, 4}) <br/>
    
  Input | CBR_L1 | CBR_L2 | CBR_L3 | CBR_L4 
  --- | --- | --- | --- | --- 
  Train dataset (benign) | 0.96969697 | 0.95959596 | 1.0 | 0.95959596
  Test dataset (benign) | 0.96938776 | 0.96938776 | 0.95918367 | 1.0
  Test dataset (adv) | 0.10638298 | 0.57446809 | 0.25531915 | 0.9893617
    
  <img src="../Images/Exp9/exp9_96.png" align="center" border="0" width="576" height="384"/>
  </div>
  <br/>
  <div>
  CBR in LP_i layer with <b>qr=0.97</b> (i={1, 2, 3, 4}) <br/>
    
  Input | CBR_L1 | CBR_L2 | CBR_L3 | CBR_L4 
  --- | --- | --- | --- | --- 
  Train dataset (benign) | 0.96969697 | 0.96969697 | 1.0 | 0.96969697
  Test dataset (benign) | 0.96938776 | 0.96938776 | 0.98979592 | 1.0
  Test dataset (adv) | 0.10638298 | 0.60638298 | 0.27659574 | 1.0
    
  <img src="../Images/Exp9/exp9_97.png" align="center" border="0" width="576" height="384"/>
  </div>
  <br/>
  <div>
  CBR in LP_i layer with <b>qr=0.98</b> (i={1, 2, 3, 4}) <br/>
    
  Input | CBR_L1 | CBR_L2 | CBR_L3 | CBR_L4 
  --- | --- | --- | --- | --- 
  Train dataset (benign) | 0.98989899 | 0.97979798 | 1.0 | 0.97979798
  Test dataset (benign) | 0.98979592 | 0.97959184 | 0.98979592 | 1.0
  Test dataset (adv) | 0.11702128 | 0.76595745 | 0.27659574 | 1.0
    
  <img src="../Images/Exp9/exp9_98.png" align="center" border="0" width="576" height="384"/>
  </div>
  <br/>
  <div>
  CBR in LP_i layer with <b>qr=0.99</b> (i={1, 2, 3, 4}) <br/>
    
  Input | CBR_L1 | CBR_L2 | CBR_L3 | CBR_L4 
  --- | --- | --- | --- | --- 
  Train dataset (benign) | 0.98989899 | 0.98989899 | 1.0 | 0.98989899
  Test dataset (benign) | 0.98979592 | 0.98979592 | 0.98979592 | 1.0
  Test dataset (adv) | 0.12765957 | 0.85106383 | 0.27659574 | 1.0
    
  <img src="../Images/Exp9/exp9_99.png" align="center" border="0" width="576" height="384"/>
  </div>
  <br/>
  <div>
  CBR in LP_i layer with <b>qr=1.00</b> (i={1, 2, 3, 4}) <br/>
    
  Input | CBR_L1 | CBR_L2 | CBR_L3 | CBR_L4 
  --- | --- | --- | --- | --- 
  Train dataset (benign) | 1.0 | 1.0 | 1.0 | 1.0
  Test dataset (benign) | 1.0 | 1.0 | 1.0 | 1.0
  Test dataset (adv) | 0.13829787 | 0.94680851 | 0.29787234 | 1.0
    
  <img src="../Images/Exp9/exp9_100.png" align="center" border="0" width="576" height="384"/>
  </div>
 