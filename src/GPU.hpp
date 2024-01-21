#ifndef GPU_HPP
#define GPU_HPP

#include <webgpu/webgpu_cpp.h>

class GPU {
 public:
  GPU();

  wgpu::Instance& Instance() { return instance_; }
  wgpu::Device& Device() { return device_; }

 public:
  void OnAdapterFound(WGPURequestAdapterStatus status,
                      WGPUAdapter adapter_handle,
                      char const* message);
  void OnDeviceFound(WGPURequestDeviceStatus status,
                     WGPUDevice device_handle,
                     char const* message);
  void OnError(WGPUErrorType type, char const* message);

 private:
  wgpu::Instance instance_;
  wgpu::Device device_;
  wgpu::Adapter adapter_;
};

#endif // GPU_HPP
