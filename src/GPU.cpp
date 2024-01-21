#include "GPU.hpp"
#include <iostream>

namespace cGPU {

void OnAdapterFound(WGPURequestAdapterStatus status,
                    WGPUAdapter adapter,
                    char const* message,
                    void* userdata) {
  GPU* gpu = reinterpret_cast<GPU*>(userdata);
  gpu->OnAdapterFound(status, adapter, message);
}

void OnDeviceFound(WGPURequestDeviceStatus status,
                   WGPUDevice device_handle,
                   char const* message,
                   void* userdata) {
  GPU* gpu = reinterpret_cast<GPU*>(userdata);
  gpu->OnDeviceFound(status, device_handle, message);
}

void OnError(WGPUErrorType type, char const* message, void* userdata) {
  GPU* gpu = reinterpret_cast<GPU*>(userdata);
  gpu->OnError(type, message);
}

void OnDeviceLost(WGPUDeviceLostReason reason,
                  char const* message,
                  void* userdata) {
  std::cout << "Device lost" << std::endl;
  if (message) {
    std::cout << " - message: " << message << std::endl;
  }
}

}  // namespace cGPU

GPU::GPU() {
  instance_ = wgpu::CreateInstance();
  wgpu::RequestAdapterOptions options{
      .powerPreference = wgpu::PowerPreference::HighPerformance,
  };
  instance_.RequestAdapter(&options, cGPU::OnAdapterFound,
                           reinterpret_cast<void*>(this));
}

void GPU::OnAdapterFound(WGPURequestAdapterStatus status,
                         WGPUAdapter adapter_handle,
                         char const* message) {
  if (status != WGPURequestAdapterStatus_Success) {
    std::cout << "Failed to find an adapter: " << message << std::endl;
    exit(0);
    return;
  }

  adapter_ = wgpu::Adapter::Acquire(adapter_handle);

  wgpu::AdapterProperties properties;
  adapter_.GetProperties(&properties);
  std::cout << "GPU:" << std::endl;
  std::cout << "- vendorName: " << properties.vendorName << std::endl;
  std::cout << "- architecture: " << properties.architecture << std::endl;
  std::cout << "- name: " << properties.name << std::endl;
  std::cout << "- driverDescription: " << properties.driverDescription
            << std::endl;

  wgpu::DeviceDescriptor device_descriptor{
      .label = "neural-webgpu device",
      .deviceLostCallback = cGPU::OnDeviceLost,
      .deviceLostUserdata = nullptr,
  };
  adapter_.RequestDevice(&device_descriptor, cGPU::OnDeviceFound,
                         reinterpret_cast<void*>(this));
}

void GPU::OnDeviceFound(WGPURequestDeviceStatus status,
                        WGPUDevice device_handle,
                        char const* message) {
  if (status != WGPURequestDeviceStatus_Success) {
    std::cout << "Failed to create device: " << message << std::endl;
    exit(0);
    return;
  }

  device_ = wgpu::Device::Acquire(device_handle);

  // Add an error callback for more debug info
  device_.SetUncapturedErrorCallback(cGPU::OnError,
                                     reinterpret_cast<void*>(this));
}

void GPU::OnError(WGPUErrorType type, char const* message) {
  std::cout << "Device error: " << std::endl;
  std::cout << "- type " << type << std::endl;
  if (message) {
    std::cout << " - message: " << message << std::endl;
  }
}
