import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import disarray
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score

ALL_METRICS = [
    "accuracy",
    "f1",
    "false_discovery_rate",
    "false_negative_rate",
    "false_positive_rate",
    "negative_predictive_value",
    "positive_predictive_value",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "true_negative_rate",
    "true_positive_rate",
]

def toOneHot(arr):
    num_class = np.max(arr) + 1
    print(num_class)
    arr_list = []
    for i in range(len(arr)):
        label = [0] * num_class
        label[arr[i]] = 1
        arr_list.append(label)
    return np.array(arr_list)

def get_weighted(value_list, class_weight):
    value = 0
    for i in range(len(value_list)):
        value+=value_list[i] * class_weight[i]
    return value

def get_confusion_matrix(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    cm = metrics.confusion_matrix(y_true, y_pred)
    return cm

def get_accuracy(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def get_kappa_score(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    kappa = cohen_kappa_score(y_true, y_pred)
    return kappa

def get_metrics_for_each_class(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    cm = metrics.confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm, dtype=int)
    all_metrics=df.da.export_metrics()
    cols = all_metrics.columns
    idx = all_metrics.index
    
    classes = np.unique(y_true)
    class_cnts = [np.count_nonzero(y_true == e) for e in classes]
    class_weight = np.array(class_cnts) / (1.0*len(y_true))
    macro_list = []
    weighted_list = []
    for idx_name in idx:
        record = pd.DataFrame(all_metrics, index=[idx_name])
        value_list=[]
        for column in cols[:-1]:
            value = record[column].values[0]
            value_list.append(value)
        macro_value = np.mean(value_list)
        weighted_value = get_weighted(value_list, class_weight)
        macro_list.append(macro_value)
        weighted_list.append(weighted_value)
        
    all_metrics['macro-average'] = macro_list
    all_metrics['weighted-average'] = weighted_list
    
    weighted_AUC = roc_auc_score(y_true, y_prob, average="weighted", multi_class="ovr")
    # micro_AUC = roc_auc_score(y_true, y_prob, average="micro", multi_class="ovr")
    macro_AUC = roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr")
    
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    _y_true = np.array(toOneHot(y_true))
    _y_prob = np.array(y_prob)
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(_y_true[:, i], _y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(_y_true.ravel(), _y_prob.ravel())
    roc_auc["micro-average"] = auc(fpr["micro"], tpr["micro"])
    # roc_auc['micro-average'] = micro_AUC
    roc_auc['macro-average'] = macro_AUC
    roc_auc['weighted-average'] = weighted_AUC
    new_row = []
    for col in cols:
        new_row.append(roc_auc[col])
    new_row.append(roc_auc["macro-average"])
    new_row.append(roc_auc["weighted-average"])
    all_metrics.loc['auc'] = new_row
    return all_metrics
        
if __name__ == '__main__':
    y_true = [0,0,0,0, 1,1,1, 2,2,2,2,2,2]
    y_prob = [[0.7,0.2,0.1],[0.8,0.1,0.1],[0.2,0.6,0.2],[0.8,0.15,0.05],
          [0.2,0.7,0.1],[0.6,0.3,0.1],[0.03,0.9,0.07],
          [0.6,0.3,0.1],[0.1,0.1,0.8],[0.05,0.85,0.1],[0.03,0.07,0.9],[0.15,0.2,0.65],[0.1,0.1,0.8]]
    metric_name = 'accuracy'
    class_name = 0
    toFile = 'metrics_example.txt'
    w_id = open(toFile,'a')
    w_id.write('Epoch 0:\n')
    w_id.close()
    metrics = get_metrics_for_each_class(y_true, y_prob)
    print(metrics)
    metrics.to_csv(toFile,mode='a')
    class_names = metrics.columns
    metric_names = metrics.index
    print(metric_names)
    assert metric_name in metric_names
    assert class_name in class_names
    a = metrics.iloc[ALL_METRICS.index(metric_name)][class_name]
    print(a)
    