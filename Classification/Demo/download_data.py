import pandas as pd

# Đường dẫn đến dữ liệu trên UCI Machine Learning Repository
url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"

# Đọc dữ liệu vào DataFrame
data = pd.read_csv(url, header=None)

# Lưu dữ liệu vào file CSV
data.to_csv("kddcup99_data.csv", index=False)

print("Data saved successfully.")
