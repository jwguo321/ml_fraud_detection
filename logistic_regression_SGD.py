import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,auc, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split,GridSearchCV, ParameterGrid
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# turn debug mode on or off
DEBUG = False
# let random state fixed
RANDOM_STATE = 42

# display data information
def data_info(df):
    data = pd.DataFrame()
    data['column'] = df.columns.tolist()
    data['non_null_number'] = df.count().values
    data['null_number'] = df.isnull().sum().values
    data['null_percent'] = data['null_number'] * 100 / df.shape[0]
    data['dtype'] = df.dtypes.values
    data.set_index('column', inplace=True)
    return data

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

# read train and test data files
train_identity = pd.read_csv('train_identity.csv')
train_transaction = pd.read_csv('train_transaction.csv')
test_identity = pd.read_csv('test_identity.csv')
test_transaction = pd.read_csv('test_transaction.csv')
print("train_identity shape: ", train_identity.shape)
print("train_transaction shape: ", train_transaction.shape)
print("test_identity shape: ", test_identity.shape)
print("test_transaction shape: ", test_transaction.shape)

# join the data together
test_identity.columns = train_identity.columns.tolist()
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
print("train shape: ", train.shape)
print("test shape: ", test.shape)

# exploring data
data_detail=data_info(train).T
print(data_detail)

# generate categorical feature list
categorical_features = ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
                        'DeviceType', 'DeviceInfo']
categorical_features += ['card' + str(i) for i in range(1,7)]
categorical_features += ['M' + str(i) for i in range(1,10)]
categorical_features += ['id_' + str(i) for i in range(12,39)]
print(categorical_features)
# dealing with categorical values
for f in categorical_features:
    train[f], uniques = pd.factorize(train[f])
    if DEBUG:
        print(uniques)
    test[f], uniques = pd.factorize(test[f])

# pre-processing data
x_train = train.drop(['TransactionID', 'isFraud'], axis=1);
x_test = test.drop(['TransactionID'],axis=1)
y_train = train['isFraud']
# split the data into train set and valid set
# split strategy: Because the dataset is ordered in timestamp order,
#                 we split first 80% as train set and last 20% as valid set
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, shuffle=False,random_state=RANDOM_STATE)
print("X_train shape: ", x_train.shape)
print("Y_train shape: ", y_train.shape)
print("X_valid shape: ", x_valid.shape)
print("Y_valid shape: ", y_valid.shape)

# -------------------------------------------------------------#
# use LightGBM to train the model
params = {}
params['learning_rate'] = 0.05
# perform a binary log loss classification(logistic regression)
params['objective'] = 'binary'
params['metric'] = 'auc'
# used to speed up training and deal with over-fitting
params['bagging_fraction'] = 0.8
params['bagging_freq'] = 1
params['feature_fraction'] = 0.8
# increase general power(deal with over-fitting)
params['max_bin'] = 127
# set positive label's weight
params['scale_pos_weight'] = 4
lgb_train = lgb.Dataset(x_train, y_train)
lgb_valid  = lgb.Dataset(x_valid, y_valid)
clf = lgb.train(params,train_set=lgb_train,num_boost_round=2000,valid_sets=[lgb_train,lgb_valid],verbose_eval=100)
y_predict_train = clf.predict(x_train)
y_predict_valid = clf.predict(x_valid)

# plot results on train set
roc_auc = plot_roc_cm(y_train, y_predict_train)
print("Train auc score: ",roc_auc)
# plot results on valid set
roc_auc = plot_roc_cm(y_valid, y_predict_valid)
print("Test auc score: ", roc_auc)

# use sci-kit learn's SGDClassifier to implement Logistic Regression
# Differed from LightGBM, the sci-kit learn library cannot deal with NaN value automatically
# we need to handle NaN values by hand
x_train = x_train.fillna(-5.1)
x_valid = x_valid.fillna(-5.1)

verbose = 0
# turn on verbose mode when debug mode is on
if DEBUG:
    verbose = 1
# scale the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.fit_transform(x_valid)

# ---------------------------------------#
# Use Grid Search cv to find best parameters
grid_search = False
if grid_search:
    grid = {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],  # learning rate
        'max_iter': [1000],  # number of epochs
        'penalty': ['l2','l1'],
        'n_jobs': [-1],
        'loss': ['log'],
        'random_state': [RANDOM_STATE],
        'verbose': [1],
        'class_weight': ['balanced', 'None']
    }
    clf = SGDClassifier()
    gsearch =GridSearchCV(clf,param_grid=grid,scoring='roc_auc',cv=3)
    gsearch.fit(x_scaled,y_train)
    best_parameters = gsearch.best_estimator_.get_params()
    cv_results = gsearch.cv_results_
    print(best_parameters)
    print(cv_results)

# ------------------------------------------- #
# Logistic Regression with SGD
clf = SGDClassifier(alpha=0.001,max_iter=1000, loss='log', random_state=RANDOM_STATE,
                    class_weight='balanced', verbose = verbose
                    )
clf.fit(x_scaled, y_train)

y_train_predict = clf.predict_proba(x_scaled)
y_predict = clf.predict_proba(x_valid_scaled)
# plot results on train set
roc_auc = plot_roc_cm(y_train, y_train_predict[:,-1])
print("Train auc score: ", roc_auc)

# plot results on valid set
roc_auc = plot_roc_cm(y_valid, y_predict[:,-1])
print("Valid auc score: ", roc_auc)