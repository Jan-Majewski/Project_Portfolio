
<!-- PROJECT LOGO -->


<br />
<p align="center">
  <a href="https://github.com/Jan-Majewski/Project_Portfolio/02_Default_prediction">
    <img src="logo.png" alt="Logo" width="591" height="332">
  </a>



</p>

-->

# Credit default prediction  - ML models comparison




<!-- Add buttons here -->

![GitHub last commit](https://img.shields.io/github/last-commit/Jan-Majewski/Project_Portfolio?02_Default_prediction)
[![LinkedIn][linkedin-shield]][linkedin-url]




<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Key Takeaways](#key-takeaways)



<!-- ABOUT THE PROJECT -->
## About The Project

This project was part of my Master Thesis in Finance and Accounting, in which I compared several approaches to default prediction with use of ML models. The analyzed dataset consist of several k clients of a loan company. As the models' output is aimed to be used to locate client's, who are most likely not to pay the next installment, we will try to predict which clients are most likely to have payment completeness (actual payment/installment value) close to 0%. To simplify model comparison I focused on predicting default during 10th payment based on history of previous 9 payments and clientS' characteristic. 


#### The project consists of 3 parts:

* PART 1) Data transformation and EDA - transforming and analyzing source data, feature engineering and visualizing Credit Portfolio characteristic with plotly. 

* PART 2) Basic ML models - comparing performance of basic ML models, Annomaly Detection algorithms and DNN. 
Used models: logistic regression, Random Forest, SVM, Ensemble of multiple models, anomaly detection with Isolation Forest and Autoencoders, Deep Neural Network. Model regularization with use hyperparameter tuning based on cross-validation and  grid search. Investigating Bayes Unavoidable Error. 

* PART 3) Recurrent Neural Networks - Focus on different architectures of RNNs. Comparison of single and multilayer RNNs, use of LSTM and GRU cells. Comparing model performance and client segmentation based on default risk based on model prediction. Analyzing feasibility of using models in production environment. 


### Built With

* Plotly
* SciKit-learn
* TensorFlow/Keras



### Key takeaways

RNN model with GRU cells turned out to be most effective in predicting credit default based on historic payments of each client. Most effective models were able to correctly flag nearly 50% of all defaults within the most risky 20% of all clients in test set, which exceeded expectations in such unpredictable environment and limited data about clients. 

Client behaviour during payment of previous installements turned out to provide more information about the default risk than the client characteristic intself. 



[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/jan-majewski-132907104/
