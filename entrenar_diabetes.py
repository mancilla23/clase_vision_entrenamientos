import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

print(tf.config.list_physical_devices())

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

data = pd.read_csv("database_diabetes/diabetes.csv")

print(data.head())
print(data.columns)

X = data.drop("Outcome", axis=1).values.astype(np.float32)
y = data["Outcome"].values.astype(np.int32)

print("Distribución clases:", np.bincount(y))

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=seed
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


classes = np.array([0, 1])
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {0: float(weights[0]), 1: float(weights[1])}


def create_model(input_dim):

    inputs = tf.keras.Input(shape=(input_dim,))

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(inputs)

    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = tf.keras.layers.Dropout(0.20)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)

model = create_model(X_train.shape[1])

metrics = [
    tf.keras.metrics.AUC(curve="ROC", name="auc_roc"),
    tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.Precision(name="precision"),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=metrics
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc_roc",
        mode="max",
        patience=20,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

pred_test = model.predict(X_test).ravel()

auc_roc = roc_auc_score(y_test, pred_test)
auc_pr = average_precision_score(y_test, pred_test)

print("AUC-ROC:", auc_roc)
print("AUC-PR:", auc_pr)


thresholds = np.linspace(0.05, 0.95, 50)

def evaluate_threshold(t):

    y_pred = (pred_test >= t).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)

    return cm, precision, recall


target_recall = 0.80
best_threshold = 0.5
best_precision = -1

recall_values = []
precision_values = []

for t in thresholds:

    cm, precision, recall = evaluate_threshold(t)

    recall_values.append(recall)
    precision_values.append(precision)

    if recall >= target_recall and precision > best_precision:
        best_precision = precision
        best_threshold = t

cm, precision, recall = evaluate_threshold(best_threshold)

tn, fp, fn, tp = cm.ravel()

print("Threshold:", best_threshold)
print("Confusion Matrix:\n", cm)
print("Recall:", recall)
print("Precision:", precision)

y_pred = (pred_test >= best_threshold).astype(int)

print(classification_report(
    y_test,
    y_pred,
    target_names=["No Diabetes", "Diabetes"]
))

plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.legend(["Train", "Validation"])
plt.show()

plt.figure()
plt.plot(history.history["auc_roc"])
plt.plot(history.history["val_auc_roc"])
plt.title("AUC ROC")
plt.legend(["Train", "Validation"])
plt.show()

plt.figure()
plt.plot(history.history["recall"])
plt.plot(history.history["val_recall"])
plt.title("Recall")
plt.legend(["Train", "Validation"])
plt.show()

fpr, tpr, _ = roc_curve(y_test, pred_test)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.title(f"ROC Curve (AUC={auc_roc:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

precision_curve, recall_curve, _ = precision_recall_curve(y_test, pred_test)

plt.figure()
plt.plot(recall_curve, precision_curve)
plt.title(f"Precision-Recall Curve (AUC={auc_pr:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["No Diabetes", "Diabetes"])
plt.yticks([0,1], ["No Diabetes", "Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

plt.figure()
plt.plot(thresholds, recall_values)
plt.title("Recall vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Recall")
plt.show()