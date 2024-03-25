from datetime import datetime
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.python.keras import layers, models

# Đọc dữ liệu từ tập tin CSV
file_path = os.getcwd() + "/data/kddcup99_data.csv"
data = pd.read_csv(file_path)

# Xem thông tin cơ bản của dữ liệu
print("Shape of the data:", data.shape)
print("Columns:", data.columns)

# Tiền xử lý dữ liệu
# Chuyển đổi các biến categorical thành dạng số
label_encoder = LabelEncoder()
data['protocol_type'] = label_encoder.fit_transform(data['protocol_type'])
data['service'] = label_encoder.fit_transform(data['service'])
data['flag'] = label_encoder.fit_transform(data['flag'])
data['label'] = label_encoder.fit_transform(data['label'])

# Chia dữ liệu thành features (X) và label (y)
X = data.drop(columns=['label'])
y = data['label']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape dữ liệu cho CNN input
X_train = np.array(X_train).reshape((-1, X_train.shape[1], 1))
X_test = np.array(X_test).reshape((-1, X_test.shape[1], 1))

# Xây dựng mô hình CNN
model = models.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Lưu mô hình ra file
model.save(os.getcwd() + "/model/result/trained_cnn_{}.h5".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

# Dự đoán trên tập kiểm tra
y_pred = model.predict_classes(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", matrix)
