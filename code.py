# import libraries
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
# train
def train(train_dataset):
# split train_dataset and verify_dataset from train_dataset
    kfold = StratifiedKFold(n_splits = 60, shuffle = False)
    features = train_dataset.drop(["id", "label"], axis = 1)[:50000]
    labels = train_dataset["label"][:50000]
    for fold_num, (train_index, varify_index) in enumerate(kfold.split(features, labels)):
        feature_train = features.iloc[train_index,:]
        label_train = labels.iloc[train_index]
        feature_varify = features.iloc[varify_index,:]
        label_varify = labels.iloc[varify_index]
        model = LGBMClassifier()
        model.fit(feature_train, label_train)
        label_pred = model.predict_proba(feature_varify)[:, 1]
        auc = roc_auc_score(label_varify, label_pred)
        print("fold {:d}:auc = {:.4f}".format(fold_num, auc))
    return model
# test
def test(model, test_dataset):
    feature_test = test_dataset.drop(["id"], axis = 1)
    label_pred = model.predict(feature_test)
    print(label_pred.shape)
    submittion = pd.read_csv("submit_example_A.csv")
    submittion["label"] = label_pred
    submittion.to_csv("submittion.csv", index = False)

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
model = train(train_dataset = train_data_label)
test(model, test_dataset = test_data)
