import os
import sys
import pandas as pd
import numpy as np

MAX_BYTE_LENGTH = 224
np.set_printoptions(threshold=30)

def extract_flows(df):
    # Xác định các flow dựa trên IP và port từ DataFrame
    flows = {}
    for _, packet in df.iterrows():
        key = (packet['ip_src'], packet['port_src'], packet['ip_dst'], packet['port_dst'])
        if key not in flows:
            flows[key] = []
        flows[key].append(packet['data'])

    # Tạo ma trận cho mỗi flow
    flow_matrices = []
    for key, data_list in flows.items():
        matrix = None
        row_count = 0
        for data in data_list:
            byte_array = bytes.fromhex(data)
            if len(byte_array) >= MAX_BYTE_LENGTH:
                int_array = np.frombuffer(byte_array[:MAX_BYTE_LENGTH], dtype=np.uint8).astype(int)  # Chuyển đổi thành số nguyên
                if matrix is None:
                    matrix = np.array([int_array])
                else:
                    if row_count < 20:
                        matrix = np.vstack([matrix, int_array])
                    else:
                        # Nếu ma trận đã đủ 20 dòng thì thêm vào danh sách flow_matrices
                        flow_matrices.append((key, matrix))
                        matrix = np.array([int_array])
                        row_count = 0
                row_count += 1
        # Kiểm tra xem ma trận có đủ 20 dòng không, nếu không thì không thêm vào danh sách
        if matrix is not None and matrix.shape[0] == 20:
            flow_matrices.append((key, matrix))

    return flow_matrices

def extract_flows_from_directory(directory):
    df_all = pd.DataFrame()

    # Duyệt qua tất cả các thư mục con và tệp tin trong thư mục
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".csv"):
                file_path = os.path.join(root, filename)
                print(f"Đường dẫn file: {file_path}")
                # Đọc dữ liệu từ file CSV và thêm vào DataFrame
                df = pd.read_csv(file_path, nrows=100, on_bad_lines='skip')
                df_all = pd.concat([df_all, df], ignore_index=True)

    # Xử lý extract_flows từ DataFrame đã được tổng hợp
    flow_matrices = extract_flows(df_all)
    return flow_matrices

# Thực thi chương trình với thư mục chứa tất cả các file CSV
directory = "E:\\Backup\\Dataset\\tiktok"
flow_matrices = extract_flows_from_directory(directory)
for idx, (key, matrix) in enumerate(flow_matrices, start=1):
    print(f"Flow {idx}:")
    print(f"  IP source: {key[0]}, Port source: {key[1]}")
    print(f"  IP destination: {key[2]}, Port destination: {key[3]}")
    print("  Matrix:")
    print(matrix)
    print()