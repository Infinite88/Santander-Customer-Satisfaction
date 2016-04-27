# Predicting a Customer's Satisfaction

## 1. Project Overview

Which customers are happy customers?

Customer satisfaction is a key measure of success and unhappy customers don't stick around. What's more, unhappy customers rarely voice their dissatisfaction before leaving. 

Santander Bank is asking for help in predicting dissatisfied customers early in their relationship. Doing so would allow Santander to take proactive steps to improve a customer's happiness before it's too late.

This data set has hundreds of anonymized features to predict if a customer is satisfied or dissatisfied with their banking experience.

## 2. Metrics
Because trying to predict whether a customer is satisfied or unsatisfied with their service, this becomes a supervised, classification problem. The metric that is used to evaluate the model is the Area Under the Receiver Operating Characteristic (AUROC). 

>The receiver operating characteristic (ROC), or ROC curve, is a graphical plot that illustrates the performance of a binary classifier system as its discrimination threshold is varied. The curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. The true-positive rate is also known as sensitivity, or recall. The false-positive rate is also known as the fall-out and can be calculated as (1 - specificity). The ROC curve is thus the sensitivity as a function of fall-out. The advantage of the roc curve is that it explores all possible setting of the threshold.


>![alt tag](https://docs.eyesopen.com/toolkits/cookbook/python/_images/roc-theory-small.png)

## 3. Analysis

### Data Exploration
The data set appears to very clean with the exception of a lot of 0 values. The table below illustrates the data prior to being clean.

>| Table I. Data Summary |
>|-----------------------|
>|# of Entries:  | 76020|
>|# of Features: | 371 |
>|# of Satisfied | 73012 |
>|# of Unsatisfied | 3008 |


Throughout my data auditing I found that there were many duplicate features and constant features. I dealt with this by simply removing those columns from the data set. After removing the previously mention features, only the the most important features were kept. The table below illustrates the data after being cleaned.

>| Table Ia. Data Summary |
>|-----------------------|
>|# of Entries:  | 76020|
>|# of Features: | 39 |
>|# of Satisfied | 73012 |
>|# of Unsatisfied | 3008 |


Feature importance

###Algorithms and Techniques
In trying to find the best model, seven different untuned classifiers were used. The performance of each model can be seen on the table below.


>|                Table II. Model Performance                  |
>|-------------------------------------------------------------|
>|         Classifier         | Train Time (sec) | AUROC Score |
>|----------------------------|------------------|-------------|
>|         Naive Bayes        |        0.45      |    0.745    |
>|        Random Forest       |        9.68      |    0.684    |
>|        Decision Tree       |        8.25      |    0.575    |
>|     Logistic Regression    |       45.58      |    0.576    |
>|      Gradient Boosting     |      111.50      |    0.835    |
>|  Extreme Gradient Boosting |       13.55      |    0.837    |
>|          Ada Boosting      |       36.23      |    0.826    |

As mentioned previously, the AUROC score was used to measure performance. Because the data has already been split into a training and testing data set, there is no need to actually split this data. Therefore, a 10-fold cross-validation method was used. How this works is ne fold of the data set is chosen as the test set, and the rest is chosen training set. The process is repeated 10 times and the average score is reported. The running time includes the total time of training and testing time in cross-validation process.

Benchmark
As seen on the table above, the top three scores were XGB, gradient boosting, and adaboosting. With a slightly higher score than gradient boosting, the model that seemed to perform the best was XGB. Not only was the score slightly higher, but XGB was blazingly faster than GBM. 

## 4. Methodology

Now knowing what model that will be used, fine tuning the parameters would be the next step.

XGBoost parameters can be divided into three catergories:

> 1. General Parameters: Parameters that define the overall functionality of XGBoost.
> 2. Booster Parameters: Guides the individual booster (tree/regression) at each step.
> 3. Learning Task Parameters: Parameters used to define the optimization objective the metric to be calculated at each step.

To achieve the best possible model, a variety of different combinations were used to increase the AUROC score.

The below table illustrates the parameters used in the final model.

Here wew can see that the final AUROC score has increased by __

## 5. Results
Model Evaluation and Validation: use learning curve graphs
Justification

## 6. Conclusion

Reflection

I really enjoyed working on this project. I found it to very interesting in trying to predict if a customer will be satisfied or not with a company's service. With the data set being anonymized, it makes it very hard to explore what each feature actually means and to truly explain the true importance of a feature to another. I also found out that the more complexed ensemble method had better performance scores than simple classifiers. Although the tuning of the model's parameters increased the the final score, methods like feature engineering, creating ensemble of models, stacking, etc may improve the model significantly. Although this would be a good thing, it may take a long time to train.


## References
> https://www.kaggle.com/c/santander-customer-satisfaction
> http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
> https://en.wikipedia.org/wiki/Receiver_operating_characteristic

