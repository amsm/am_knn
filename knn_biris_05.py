from sklearn.datasets import load_iris #, load_digits, load_breast_cancer
import matplotlib.pyplot as plt

ds_data = load_iris()
# ds_data = load_breast_cancer()
ds_samples = ds_data.data
ds_labels = ds_data.target
ds_label_names = ds_data.target_names
ds_feature_names = ds_data.feature_names

def ver_todas_as_amostras():
    for amostra in ds_samples:
        print(amostra)
    # for
# def ver_todas_as_amostras

def ver_todas_as_amostras_e_label_correspondente():
    for amostra, label in zip(ds_samples, ds_labels):
        print(f"{amostra}: {label} ({ds_label_names[label]})")
    # for
# def ver_todas_as_amostras_e_label_correspondente

ver_todas_as_amostras_e_label_correspondente()

def visualizar_feature(
    p_feature_idx =0
):
    dados = [ (label, valor) for label, valor in zip(ds_labels, ds_samples[:,p_feature_idx]) ] # pay attention!
    plt.scatter(
        [ v[0] for v in dados ], # labels, x
        [ v[1] for v in dados ], # features values, y
        c=[ v[0] for v in dados ]
    )
    plt.xticks(
        range(len(ds_label_names)),
        ds_label_names,
    )
    plt.xlabel("Tipo de Iris")
    plt.ylabel(f"{ds_feature_names[p_feature_idx]}")
    plt.title(f"{ds_feature_names[p_feature_idx]}")
    plt.show()
# visualizar_feature

visualizar_feature(1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    ds_samples,
    ds_labels,
    test_size=0.2,
    shuffle=True, # the default
    random_state=1,
)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(
    metric='minkowski',
    n_neighbors=9
)
fit_result = knn.fit(X_train, y_train)
print(fit_result) # KNeighborsClassifier(n_neighbors=1)

test_predictions = knn.predict(X_test)
print(test_predictions)
print(y_test)

# ok, but no need
performance  = [ pred==real for pred, real in zip(test_predictions, y_test) ]

print (type(test_predictions)) # class 'numpy.ndarray'
# because both operands are numpy.ndarrays, element-wise comparison will work
performance_np = test_predictions == y_test
print (performance_np)

import numpy as np
accuracy = np.mean(performance_np)
print(accuracy)

from sklearn.metrics import confusion_matrix, \
    accuracy_score, \
    precision_score, \
    recall_score, \
    f1_score
from sklearn.metrics import classification_report

cr = classification_report(
    y_test, # the true labels
    test_predictions,
)
print(cr)

cm = confusion_matrix(
    y_test, # verdade
    test_predictions # previsões
)
print(cm)

"""
[[11  0  0]
 [ 0 13  0]
 [ 0  0  6]]
"""

# Calculate individual metrics
cm_accuracy = accuracy_score(
    y_test,
    test_predictions
)
cm_precision = precision_score(
    y_test,
    test_predictions,
    average='macro'
    #average=None
)
cm_recall = recall_score(
    y_test,
    test_predictions,
    average='macro'
    #average=None
)
cm_f1 = f1_score(
    y_test,
    test_predictions,
    average='macro'
    #average=None
)

def present_metric(
    p_measurements_array,
    p_metric_name
):
    msg = f"Metric: {p_metric_name}\n"

    for measurement, idx in zip(p_measurements_array, range(len(p_measurements_array))):
        feature_name = ds_feature_names[idx]
        msg += f"{feature_name} : {measurement}\n"
    # for
    print(msg)
# def present_metric

msg_accuracy = f"cm_accuracy={cm_accuracy}"
msg_precision = f"cm_precision={cm_precision}"
msg_recall = f"cm_recall={cm_recall}"
msg_f1 = f"cm_f1={cm_f1}"

msgs = [msg_accuracy, msg_precision, msg_recall, msg_f1]
for m in msgs:
    print(m)
# for

"""
present_metric(
    cm_precision,
    "Precision"
)
"""