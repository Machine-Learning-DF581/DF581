# import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier#效果不行
from sklearn.ensemble import GradientBoostingClassifier

# train
def train_test(train_dataset, test_dataset):
# split train_dataset and verify_dataset from train_dataset
    k = 10
    kfold = StratifiedKFold(n_splits = 10, shuffle = False)
    features = train_dataset.drop(["id", "label"], axis = 1)[:50000]
    labels = train_dataset["label"][:50000]
    feature_test = test_dataset.drop(["id"], axis = 1)
    labels_pred = np.zeros(labels.shape[0])
    test_label_pred = np.zeros(test_dataset.shape[0])
    for fold_num, (train_index, varify_index) in enumerate(kfold.split(features, labels)):
        feature_train = features.iloc[train_index,:]
        label_train = labels.iloc[train_index]
        feature_varify = features.iloc[varify_index,:]
        label_varify = labels.iloc[varify_index]
        model = GradientBoostingClassifier()
        model.fit(feature_train, label_train)
        label_pred = model.predict_proba(feature_varify)[:, 1]
        labels_pred[varify_index] = label_pred
        auc = roc_auc_score(label_varify, label_pred)
        print("fold {:d}:auc = {:.4f}".format(fold_num, auc))
        # test
        test_label_pred = test_label_pred + model.predict_proba(feature_test)[:, 1]
        if fold_num == k - 1:
            test_label_pred = test_label_pred / k
            all_auc = roc_auc_score(labels, labels_pred)
            print("all fold:auc = {:.4f}".format(all_auc))
    submission = pd.read_csv("submit_example_A.csv")
    submission["label"] = test_label_pred
    submission.to_csv("submission.csv", index = False)    

# import data
train_data_label = pd.read_csv("dataTrain.csv")
train_data_nolabel = pd.read_csv("dataNoLabel.csv")
test_data = pd.read_csv("dataA.csv")
# change "f3" from str to int
label_encoder = LabelEncoder()
data_total = pd.concat([train_data_label, train_data_nolabel, test_data])
label_encoder.fit(data_total["f3"])
train_data_label["f3"] = label_encoder.transform(train_data_label["f3"])
train_data_nolabel["f3"] = label_encoder.transform(train_data_nolabel["f3"])
test_data["f3"] = label_encoder.transform(test_data["f3"])
# create location feature 
feature_location = ["f1", "f2", "f4", "f5", "f6"]
for i in range(len(feature_location)):
    for j in range(i + 1, len(feature_location)):
        train_data_label[feature_location[i] + "+" + feature_location[j]] = train_data_label[feature_location[i]] + train_data_label[feature_location[j]]
        train_data_label[feature_location[i] + "-" + feature_location[j]] = train_data_label[feature_location[i]] - train_data_label[feature_location[j]]
        train_data_label[feature_location[i] + "*" + feature_location[j]] = train_data_label[feature_location[i]] * train_data_label[feature_location[j]]
        train_data_label[feature_location[i] + "/" + feature_location[j]] = train_data_label[feature_location[i]] / (train_data_label[feature_location[j]] + 1)
        train_data_nolabel[feature_location[i] + "+" + feature_location[j]] = train_data_nolabel[feature_location[i]] + train_data_nolabel[feature_location[j]]
        train_data_nolabel[feature_location[i] + "-" + feature_location[j]] = train_data_nolabel[feature_location[i]] - train_data_nolabel[feature_location[j]]
        train_data_nolabel[feature_location[i] + "*" + feature_location[j]] = train_data_nolabel[feature_location[i]] * train_data_nolabel[feature_location[j]]
        train_data_nolabel[feature_location[i] + "/" + feature_location[j]] = train_data_nolabel[feature_location[i]] / (train_data_nolabel[feature_location[j]] + 1)
        test_data[feature_location[i] + "+" + feature_location[j]] = test_data[feature_location[i]] + test_data[feature_location[j]]
        test_data[feature_location[i] + "-" + feature_location[j]] = test_data[feature_location[i]] - test_data[feature_location[j]]
        test_data[feature_location[i] + "*" + feature_location[j]] = test_data[feature_location[i]] * test_data[feature_location[j]]
        test_data[feature_location[i] + "/" + feature_location[j]] = test_data[feature_location[i]] / (test_data[feature_location[j]] + 1)
# create call feature 
feature_call = ["f43", "f44", "f45", "f46"]
for i in range(len(feature_call)):
    for j in range(i + 1, len(feature_call)):
        train_data_label[feature_call[i] + "+" + feature_call[j]] = train_data_label[feature_call[i]] + train_data_label[feature_call[j]]
        train_data_label[feature_call[i] + "-" + feature_call[j]] = train_data_label[feature_call[i]] - train_data_label[feature_call[j]]
        train_data_label[feature_call[i] + "*" + feature_call[j]] = train_data_label[feature_call[i]] * train_data_label[feature_call[j]]
        train_data_label[feature_call[i] + "/" + feature_call[j]] = train_data_label[feature_call[i]] / (train_data_label[feature_call[j]] + 1)
        train_data_nolabel[feature_call[i] + "+" + feature_call[j]] = train_data_nolabel[feature_call[i]] + train_data_nolabel[feature_call[j]]
        train_data_nolabel[feature_call[i] + "-" + feature_call[j]] = train_data_nolabel[feature_call[i]] - train_data_nolabel[feature_call[j]]
        train_data_nolabel[feature_call[i] + "*" + feature_call[j]] = train_data_nolabel[feature_call[i]] * train_data_nolabel[feature_call[j]]
        train_data_nolabel[feature_call[i] + "/" + feature_call[j]] = train_data_nolabel[feature_call[i]] / (train_data_nolabel[feature_call[j]] + 1)
        test_data[feature_call[i] + "+" + feature_call[j]] = test_data[feature_call[i]] + test_data[feature_call[j]]
        test_data[feature_call[i] + "-" + feature_call[j]] = test_data[feature_call[i]] - test_data[feature_call[j]]
        test_data[feature_call[i] + "*" + feature_call[j]] = test_data[feature_call[i]] * test_data[feature_call[j]]
        test_data[feature_call[i] + "/" + feature_call[j]] = test_data[feature_call[i]] / (test_data[feature_call[j]] + 1)
# train and test
train_test(train_dataset = train_data_label, test_dataset = test_data)
