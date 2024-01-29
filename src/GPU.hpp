#ifndef GPU_HPP
#define GPU_HPP

#include <webgpu/webgpu_cpp.h>
#include <string>

class GPU {
 public:
  GPU();
  ~GPU();

  wgpu::Instance& Instance() { return instance_; }
  wgpu::Device& Device() { return device_; }

  const std::string& VendorName() { return vendor_name_; }
  const std::string& Architecture() { return architecture_; }
  const std::string& Name() { return name_; }
  const std::string& DriverDescription() { return driver_description_; }

 public:
  void OnAdapterFound(WGPURequestAdapterStatus status,
                      WGPUAdapter adapter_handle,
                      char const* message);
  void OnDeviceFound(WGPURequestDeviceStatus status,
                     WGPUDevice device_handle,
                     char const* message);
  void OnError(WGPUErrorType type, char const* message);
  void OnDeviceLost(WGPUDeviceLostReason reason, char const* message);

 private:
  wgpu::Instance instance_;
  wgpu::Device device_;
  wgpu::Adapter adapter_;

  std::string vendor_name_;
  std::string architecture_;
  std::string name_;
  std::string driver_description_;
};

#endif // GPU_HPP
