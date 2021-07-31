# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 05:44:48 2021

@author: 6yang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression

#import all csv files
train_trans=pd.read_csv("train_transaction.csv")
test_trans=pd.read_csv("test_transaction.csv")
train_iden = pd.read_csv('train_identity.csv')
test_iden = pd.read_csv('test_identity.csv')


#merge transactions.csv and identity.csv for traning
train_set = pd.merge(train_trans, train_iden, how='left')

#isFraud distribution in train_set
train_set['isFraud'].value_counts().plot.bar()

#Rename the id-x column in test set to id_x before merge test datasest
id_num = [i for i in test_iden.columns if i[0]+i[1] == 'id']
rename_id = {i:'id_'+str(i[-2]+i[-1]) for i in id_num}
test_iden = test_iden.rename(columns=rename_id)
test_set = test_trans.merge(test_iden,on=['TransactionID'],how='left')

#See the missing rate for each column
train_cols = train_set.columns

missing_rate=[]
for col in train_cols:
    if col != 'isFraud':
        df = pd.concat([train_set[col],test_set[col]],axis=0)
        missing = round((df.isnull().sum()/df.shape[0])*100,2)
        missing_rate.append(missing)
        print(f'missing rate for {col} is {missing}%')
        
#preprocess
#Change data type for columns
cat_cols = (['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
             'addr1', 'addr2', 
             'P_emaildomain', 'R_emaildomain', 
             'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 
             'DeviceType', 'DeviceInfo', 
             'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'])

type_map = {c: str for c in cat_cols}
train_set[cat_cols] = train_set[cat_cols].astype(type_map, copy=False)
test_set[cat_cols] = test_set[cat_cols].astype(type_map, copy=False)
id_cols = ['TransactionID', 'TransactionDT']

#Original train and test set
y_train_ = train_set['isFraud']
X_train = train_set.drop(columns=['isFraud'])
X_test = test_set.copy()

#Uncomment following del code to save RAM if necessary
#del train_trans, train_iden
#del test_trans, test_iden
#del test_set, train_set

#Min max scaler to sclae data
for col in X_train.columns:
    if col in cat_cols:
        dff = pd.concat([X_train[col],X_test[col]])
        dff,_ = pd.factorize(dff,sort=True)
        if dff.max()>32000: 
            print(col,'needs int32 datatype')
            
        X_train[col] = dff[:len(X_train)].astype('int16')
        X_test[col] = dff[len(X_train):].astype('int16')
cols = X_train.columns
for col in cols:
    if col not in cat_cols and col not in id_cols:
        # min max scalar
        dff = pd.concat([X_train[col],X_test[col]])
        dff = (dff - dff.min())/(dff.max() - dff.min())
        dff.fillna(-1,inplace=True)

        X_train[col] = dff[:len(X_train)]
        X_test[col] = dff[len(X_train):]
     
#Split train dataset to train and validation dataset
split=int(X_train.shape[0]*0.8)
train_index = X_train.index[:split]  
validation_index = X_train.index[split:]
x_train = X_train.iloc[train_index]
y_train = y_train_.iloc[train_index]
x_cv_ = X_train.iloc[validation_index]
y_cv = y_train_.iloc[validation_index]

#Uncomment following del code to save RAM if necessary
#del X_train,X_test

#Build and train the model
model = RandomForestClassifier()
model.fit(x_train,y_train)

# plot roc_auc curve and confusion matrix
def plot_roc_cm(y_true, y_score):
    # plot confusion matrix
    y_predict_label = [0 if y < 0.5 else 1 for y in y_score]
    cmat = confusion_matrix(y_true, y_predict_label)
    cm = pd.DataFrame(cmat, columns=np.unique(y_true), index = np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    plt.figure()
    sns.heatmap(cm, cmap='Blues',annot=True,annot_kws={"size": 15}, fmt='g')
    plt.show()
    # plot roc curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return roc_auc

#Get predict proba to plot roc curve and confusion matrix 
y_train_predprob = model.predict_proba(x_train)
y_test_predprob = model.predict_proba(x_cv_)

# plot results on train set
roc_auc = plot_roc_cm(y_train, y_train_predprob[:,-1])
print("Train auc score: ", roc_auc)

# plot results on valid set
roc_auc = plot_roc_cm(y_cv, y_test_predprob[:,-1])
print("Valid auc score: ", roc_auc)


