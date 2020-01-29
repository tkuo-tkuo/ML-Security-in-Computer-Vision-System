
Note that all mathematical formulas are written for <b>only a layer</b> in a given neural network instead of the whole neural network.  

<details>
  
  <summary>Original Rule-based Method: Determine whether the provenance of input is learned</summary>
  
  <img src="../README_images/original_method.png" align="center" border="0" width="900" height="121"/>

</details>

<details>
  
  <summary>Potential Method 1: Compute SP (Sample Probability) for each neuron & Determine whether l1-distance between SP(LP) and provenance of input is close enough</summary>
  
  <img src="../README_images/potential_method_1.png" align="center" border="0" width="900" height="121"/>

</details>

<details>
  
  <summary>Potential Method 2: Compute SP (Sample Probability) for each neuron & Determine whether l1-distance between SP(LP) and provenance of input is close enough, where we filter out neurons that is relatively close (< beta).</summary>
  
  <img src="../README_images/potential_method_2.png" align="center" border="0" width="900" height="226"/>

</details>

<details>
  
  <summary>Potential Method 3: Compute the probability that each neuron to be benign & Multiple all probabilities of neurons to determine whether a given input is benign/adversarial by a probability threshold value.</summary>

  <img src="../README_images/potential_method_3.png" align="center" border="0" width="900" height="272"/>
  
</details>

<details>
  
  <summary>Potential Method 4: Compute the probability that each neuron to be benign & Multiple all probabilities of neurons to determine whether a given input is benign/adversarial by a probability threshold value, where we filter out neurons with probabilities relatively ambigious (e.g., 0.3 - 0.7)</summary>

  <img src="../README_images/potential_method_4.png" align="center" border="0" width="900" height="343"/>

</details>
