import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

def process_data(file_path, label):
    print(f"Đang xử lý file: {file_path}")  
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(file_path)

    # Kiểm tra xem file có dữ liệu không
    if df.empty:
        print(f"File {file_path} trống.")
        return pd.DataFrame()  

    print(f"Đọc {len(df)} dòng dữ liệu từ file {file_path}.")

    # Tính toán các feature
    features = {
        'packet_count': [],
        'average_packet_length': [],
        'average_inter_packet_time': [],
        'flow_duration': [],
    }

    # Đặt cửa sổ thời gian (time window)
    time_window = 1000.0  
    start_time = df['frame.time_epoch'].iloc[0]
    df['frame.time_epoch'] = df['frame.time_epoch'] * 1000 #giá trị nhỏ quá nhân lên 1000 lần
    
    # Danh sách lưu trữ các gói tin trong cửa sổ
    window_packets = []

    for index, row in df.iterrows():
        # Thêm gói tin vào cửa sổ
        window_packets.append(row)

        # In ra số lượng gói tin trong cửa sổ và thời gian
        # print(f"Gói tin: {index + 1}, Thời gian: {row['frame.time_epoch']}, Thời gian bắt đầu: {start_time}")

        # Kiểm tra thời gian của gói tin đầu tiên trong cửa sổ
        if row['frame.time_epoch'] - start_time > time_window:
            # Tính toán các feature cho cửa sổ
            packet_lengths = [pkt['frame.len'] for pkt in window_packets]
            inter_packet_times = np.diff([pkt['frame.time_epoch'] for pkt in window_packets])
            
            features['packet_count'].append(len(window_packets))
            features['average_packet_length'].append(np.mean(packet_lengths))
            features['average_inter_packet_time'].append(np.mean(inter_packet_times) if len(inter_packet_times) > 0 else 0)
            features['flow_duration'].append(window_packets[-1]['frame.time_epoch'] - window_packets[0]['frame.time_epoch'])

            # Đặt lại cửa sổ
            window_packets = []
            start_time += time_window  # Cập nhật start_time

    # Tạo DataFrame cho các feature
    feature_df = pd.DataFrame(features)

    # Thêm nhãn vào DataFrame
    feature_df['label'] = label

    print(f"Tạo {len(feature_df)} dòng dữ liệu feature cho file {file_path}.")
    return feature_df

# Đường dẫn đến thư mục chứa các file CSV
directory_path = 'D:/Dataset/ThuThap'  

# Danh sách lưu trữ DataFrame cho tất cả các file
all_features = []

# Duyệt qua các file CSV trong thư mục
for file_name in os.listdir(directory_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(directory_path, file_name)
        label = file_name.split('.')[0]  
        feature_df = process_data(file_path, label)
        all_features.append(feature_df)

# Kết hợp tất cả các DataFrame thành một DataFrame duy nhất
final_df = pd.concat(all_features, ignore_index=True)

# Kiểm tra nếu final_df không có dữ liệu
if final_df.empty:
    print("Không có dữ liệu để huấn luyện.")
else:
    # Chia dữ liệu thành train và test
    X = final_df.drop(columns=['label'])
    y = final_df['label']

    # Chia dữ liệu thành train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Xây dựng mô hình phân loại
    print("Đang huấn luyện mô hình Random Forest với 100 cây...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Đo thời gian suy luận dự đoán
    print("Bắt đầu đo thời gian suy luận...")
    start_inference_time = time.perf_counter()
    y_pred = model.predict(X_test)
    end_inference_time = time.perf_counter()
    
    inference_time_ms = (end_inference_time - start_inference_time) * 1000
    print(f"⏱️  THỜI GIAN SUY LUẬN RANDOM FOREST: {inference_time_ms:.2f} ms")
    print(f"⏱️  Thời gian suy luận trung bình mỗi mẫu: {inference_time_ms/len(X_test):.4f} ms")

    # Đánh giá mô hình
    print("\n=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST ===")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    # Tạo và hiển thị ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    
    # Hiển thị ma trận nhầm lẫn dưới dạng text
    print("\nConfusion Matrix:")
    print(cm)
    
    # Tạo biểu đồ ma trận nhầm lẫn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    
    # Hiển thị thông tin chi tiết về các lớp
    class_names = np.unique(y)
    print(f"\nSố lượng lớp: {len(class_names)}")
    print(f"Tên các lớp: {class_names}")
    print(f"Số lượng mẫu test: {len(X_test)}")
