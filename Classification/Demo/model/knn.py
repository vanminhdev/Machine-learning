import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Đọc dữ liệu từ file CSV
path_to_data = os.getcwd() + "/data/kddcup99_data.csv"
data = pd.read_csv(path_to_data)

# Xem thông tin cơ bản của dữ liệu
print("Shape of the data:", data.shape)
print("Columns:", data.columns)

# Chia features và labels
X = data.drop(columns=["label"])  # features
y = data["label"]  # labels

# Chuyển đổi các cột dạng categorical thành dạng số
le = LabelEncoder()
X_encoded = X.apply(le.fit_transform)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Số lượng láng giềng gần nhất
knn.fit(X_train, y_train)

# Dự đoán nhãn cho dữ liệu kiểm tra
y_pred = knn.predict(X_test)

# Đánh giá độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
