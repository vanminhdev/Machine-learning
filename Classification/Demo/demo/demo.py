from datetime import datetime
import os
import numpy as np
import pandas as pd
from demo2 import my_function

my_function()

current_directory = os.getcwd()
print("Đường dẫn hiện tại:", os.path.exists(current_directory + "/data/kddcup99_data.csv"))

# Chuyển đổi sang định dạng yyyy MM dd hh mm ss
formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

print("Thời gian hiện tại: {} {}".format(formatted_time, ";"))