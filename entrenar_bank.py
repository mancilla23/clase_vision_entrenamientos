import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

# -----------------------------
# 1 Cargar dataset
# -----------------------------
data = pd.read_csv("database_bank/Churn_Modelling.csv")

print(data.head())
print("\nColumnas:", data.columns)

# eliminar columnas inútiles
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# convertir variables categóricas
data = pd.get_dummies(data)

# separar X y Y
X = data.drop("Exited", axis=1).values.astype(np.float32)
y = data["Exited"].values.astype(np.int32)

print("\nDistribución clases:", np.bincount(y))


# -----------------------------
# 2 Split train / val / test
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("\nTrain:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)


# -----------------------------
# 3 Escalamiento
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# -----------------------------
# 4 Modelo MLP
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# 5 Entrenamiento
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32
)


# -----------------------------
# 6 Graficas entrenamiento
# -----------------------------
plt.figure()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Loss durante entrenamiento")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()


plt.figure()

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Accuracy durante entrenamiento")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


# -----------------------------
# 7 Predicciones
# -----------------------------
y_prob = model.predict(X_test)

threshold = 0.7

y_pred = (y_prob > threshold).astype(int)


# -----------------------------
# 8 Matriz de confusión
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nMatriz de confusión:")
print(cm)

plt.figure(figsize=(5,5))

plt.imshow(cm, cmap="Blues")

plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")

classes = ["No Churn", "Churn"]

plt.xticks([0,1], classes)
plt.yticks([0,1], classes)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j],
                 ha="center",
                 va="center",
                 fontsize=14)

plt.colorbar()

plt.show()


# -----------------------------
# 9 Métricas
# -----------------------------
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# -----------------------------
# 10 ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.show()


# -----------------------------
# 11 Precision Recall Curve
# -----------------------------
precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall_curve, precision_curve)

plt.figure()

plt.plot(recall_curve, precision_curve, label=f"PR AUC = {pr_auc:.3f}")

plt.title("Precision Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.show()


# -----------------------------
# 12 Analizar threshold
# -----------------------------
thresholds = np.linspace(0.1,0.9,50)

precisions = []
recalls = []

for t in thresholds:

    preds = (y_prob > t).astype(int)

    precisions.append(precision_score(y_test,preds))
    recalls.append(recall_score(y_test,preds))

plt.figure()

plt.plot(thresholds,precisions,label="Precision")
plt.plot(thresholds,recalls,label="Recall")

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision vs Recall vs Threshold")

plt.legend()

plt.show()