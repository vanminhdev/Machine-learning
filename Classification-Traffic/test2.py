import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import os

# Hàm để xử lý dữ liệu từ file CSV
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
    
    # Biến lưu trữ gói tin trong một cửa sổ
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
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
