import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import glob

def read_csv_files(csv_files, mode):
    features = None
    labels = None
    
    if mode=="train":
        gt_df = pd.read_csv('gt.csv')
        labels = gt_df.iloc[:, -1]

    for file in csv_files:
        df = pd.read_csv(file)
        pred = df.iloc[:, -1]

        if features is None:
            features = pred
        else:
            features = pd.concat([features, pred], axis=1)

        

    return features, labels

train_files = glob.glob('train_csv/*.csv')
test_files = glob.glob('test_csv/*.csv')

X_train, y_train = read_csv_files(train_files, mode='train')
X_test, _ = read_csv_files(test_files, mode='test')

X_train = np.array(X_train).reshape(-1, len(train_files))
y_train = np.array(y_train)

ratio = (len(y_train)-sum(y_train)) / sum(y_train)

le = LabelEncoder()
y_train = le.fit_transform(y_train)

xgb_model = xgb.XGBClassifier(scale_pos_weight=ratio,eval_metric='aucpr')
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict_proba(X_train)

test_df = pd.read_csv(test_files[0])
new_pred = y_pred

test_df.iloc[:, -1] = new_pred[:,0].reshape(-1)

# 保存修改后的CSV文件
test_df.to_csv("ensemble.csv", index=False, header=None)
