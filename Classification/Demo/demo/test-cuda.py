import torch

# Kiểm tra xem CUDA có sẵn không
if torch.cuda.is_available():
    # Lấy số lượng GPUs
    device_count = torch.cuda.device_count()
    print(f"Có tổng cộng {device_count} GPU(s) trên hệ thống của bạn.")
    
    # Hiển thị thông tin chi tiết về từng GPU
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Di chuyển tensor lên GPU
    tensor_cpu = torch.rand(3, 3)
    tensor_gpu = tensor_cpu.cuda()
    
    # Di chuyển mô hình lên GPU
    model_cpu = torch.nn.Linear(3, 1)
    model_gpu = model_cpu.cuda()
else:
    print("CUDA không khả dụng trên hệ thống của bạn.")
