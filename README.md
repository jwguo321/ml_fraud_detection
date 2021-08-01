# COMP9417 Project
All codes used in the project are stored in `logistic_regression_SGD.py`, `Project-RF.py` and `Project-Logistic.py`. The corresponding .ipynb files stores the cache of running results.

## `Project-RF.py` and `Project-Logistic.py`
For `Project-RF.py` and `Project-Logistic.py`, it should run in Python 3 environment.
All the csv files, including train_identity.csv, train_transaction.csv, test_identity.csv, test_transaction.csv should be put in the same path with `Project-RF.py` and `Project-Logistic.py`.
To run this file, a large RAM is required. Insufficient RAM may cause the device down. To avoid this, there are some commented 'del' code, uncomment them to reduce the pressure of RAM.
`Project-RF.py` and `Project-Logistic.py` will first load all csv files, and then merge train and test dataset by the transaction ID respectively. During this process, we try to show some features of the dataset, such as the missing rate of each column and the ratio between fraud transactions and normal transactions. Then the code will do some preprocessing before train the model. Basically, we are trying to edit the data type for each column and use a simple min max scaler to sacle the dataset.
After that, we split the train dataset. We use 80% of the train dataset to train the model, and use the rest 20% for validation. In this file, we will only show the Random Forest model and Logistic Regression model. Both of them are done with sklearn package. We only shows the results with optimal parameters.
The final result will be shown in graphs, There exist a confusion matrix and a ROC curve for each result.
On our device, the Random Foerst model will take about 10 minutes and the logistic regression model will take about 20 minutes.

## `logistic_regression_SGD.py`
This python file stores following implementation of original logistic regression:
+ Logistic regression with Stochastic Gradient Descent

  We use scikit-learn's API to train the model with a SGDClassfier.
+ LightGBM

  Another framework to implement logistic regression with gradient boosting. 

## About dataset files
Since the size of dataset is too large(1.2GB in total), our team did not include them on the zip file.
These files can be download from [ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection/overview)
or google drive:

[train_transaction.csv](https://drive.google.com/file/d/1Recd-WkJnqvKkQICeq0EJdIAxLtYpZK3/view?usp=sharing)

[train_identity.csv](https://drive.google.com/file/d/1w7cL6uwuP7fZk0IWtK_sAnA2KASn3xPm/view?usp=sharing)

[test_transaction.csv](https://drive.google.com/file/d/1_vcDxwts0ANeGotvgy6azhYxvDVMeWbP/view?usp=sharing)

[test_identity.csv](https://drive.google.com/file/d/1lm7J2ZzxICYefl8mB07ttPSpofSNjw3m/view?usp=sharing)
