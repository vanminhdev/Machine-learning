import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Đọc dữ liệu từ file CSV (đảm bảo thay đường dẫn đúng vào 'path_to_data')
path_to_data = "path_to_kddcup99_data.csv"
data = pd.read_csv(path_to_data)

# Xem thông tin cơ bản của dữ liệu
print("Shape of the data:", data.shape)
print("Columns:", data.columns)

# Phân chia dữ liệu thành features và target
X = data.drop('target', axis=1)
y = data['target']

# Mã hóa các biến phân loại
le = LabelEncoder()
X_categorical = X.select_dtypes(include=['object'])
X_categorical = X_categorical.apply(le.fit_transform)

# Tiêu chuẩn hóa các biến số
scaler = StandardScaler()
X_numerical = X.select_dtypes(include=['int64'])
X_numerical = scaler.fit_transform(X_numerical)

# Ghép lại features
X = np.concatenate((X_categorical, X_numerical), axis=1)

# Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
