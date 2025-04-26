import torch

def check_cuda():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA Device: {current_device}")
        
        # Get device name
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Device Name: {device_name}")
        
        # Get CUDA version
        cuda_version = torch.version.cuda
        print(f"CUDA Version: {cuda_version}")
        
        # Get cuDNN version
        cudnn_version = torch.backends.cudnn.version()
        print(f"cuDNN Version: {cudnn_version}")
    else:
        print("CUDA is not available on this system.")

if __name__ == "__main__":
    check_cuda()