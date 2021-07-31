<H1>Project-RF.py and Project-Logistic.py</H1><br />
For Project-RF.py and Project-Logistic.py, it should run in Python 3 environment.<br />
All the csv files, including train_identity.csv, train_transaction.csv, test_identity.csv, test_transaction.csv should be put in the same path with Project-RF.py and Project-Logistic.py.<br />
To run this file, a large RAM is required. Insufficient RAM may cause the device down. To avoid this, there are some commented 'del' code, uncomment them to reduce the pressure of RAM.<br />
Project-RF.py and Project-Logistic.py will first load all csv files, and then merge train and test dataset by the transaction ID respectively. During this process, we try to show some features of the dataset, such as the missing rate of each column and the ratio between fraud transactions and normal transactions.
Then the code will do some preprocessing before train the model. Basically, we are trying to edit the data type for each column and use a simple min max scaler to sacle the dataset.<br />
After that, we split the train dataset. We use 80% of the train dataset to train the model, and use the rest 20% for validation. In this file, we will only show the Random Forest model and Logistic Regression model. Both of them are done with sklearn package. We only shows the results with optimal parameters.<br />
The final result will be shown in graphs, There exist a confusion matrix and a ROC curve for each result.<br />
On our device, the Random Foerst model will take about 10 minutes and the logistic regression model will take about 20 minutes.<br />
