#include "GPU.hpp"
#include "fmt/format.h"
#include <set>

namespace {
std::set<GPU*> g_gpus;
void AddGPU(GPU* gpu) { g_gpus.insert(gpu); }
void RemoveGPU(GPU* gpu) { g_gpus.erase(gpu); }
GPU* GetGPU(void* userdata) {
  GPU* gpu = reinterpret_cast<GPU*>(userdata);
  if (g_gpus.find(gpu) == g_gpus.end()) {
    return nullptr;
  }

  return gpu;
}


namespace cGPU {

void WithGPU(void* userdata, std::function<void(GPU*)> callback) {
  auto gpu = GetGPU(userdata);
  if (gpu) {
    callback(gpu);
  }
}

void OnAdapterFound(WGPURequestAdapterStatus status,
                    WGPUAdapter adapter,
                    char const* message,
                    void* userdata) {
  GetGPU(userdata)->OnAdapterFound(status, adapter, message);
}

void OnDeviceFound(WGPURequestDeviceStatus status,
                   WGPUDevice device_handle,
                   char const* message,
                   void* userdata) {
  GetGPU(userdata)->OnDeviceFound(status, device_handle, message);
}

void OnError(WGPUErrorType type, char const* message, void* userdata) {
  GetGPU(userdata)->OnError(type, message);
}

void OnDeviceLost(WGPUDeviceLostReason reason,
                  char const* message,
                  void* userdata) {
  WithGPU(userdata, [&](GPU* gpu) { gpu->OnDeviceLost(reason, message); });
}

}  // namespace cGPU
}  // namespace

GPU::GPU() {
  AddGPU(this);
  instance_ = wgpu::CreateInstance();
  wgpu::RequestAdapterOptions options{
      .powerPreference = wgpu::PowerPreference::HighPerformance,
  };
  instance_.RequestAdapter(&options, cGPU::OnAdapterFound,
                           reinterpret_cast<void*>(this));
}

GPU::~GPU() {
  RemoveGPU(this);
}

void GPU::OnAdapterFound(WGPURequestAdapterStatus status,
                         WGPUAdapter adapter_handle,
                         char const* message) {
  if (status != WGPURequestAdapterStatus_Success) {
    fmt::print("Failed to find an adapter: {}\n", message);
    exit(0);
    return;
  }

  adapter_ = wgpu::Adapter::Acquire(adapter_handle);

  wgpu::AdapterProperties properties;
  adapter_.GetProperties(&properties);

  vendor_name_ = properties.vendorName;
  architecture_ = properties.architecture;
  name_ = properties.name;
  driver_description_ = properties.driverDescription;

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
    fmt::print("Failed to create device: {}\n", message);
    exit(0);
    return;
  }

  device_ = wgpu::Device::Acquire(device_handle);

  // Add an error callback for more debug info
  device_.SetUncapturedErrorCallback(cGPU::OnError,
                                     reinterpret_cast<void*>(this));
}

void GPU::OnError(WGPUErrorType type, char const* message) {
  switch(type) {
    case WGPUErrorType_NoError:
      fmt::print("WGPU: No error: {}\n", message);
      break;

    case WGPUErrorType_Validation:
      fmt::print("WGPU: Validation error: {}\n", message);
      break;

    case WGPUErrorType_OutOfMemory:
      fmt::print("WGPU: Out of memory: {}\n", message);
      break;

    case WGPUErrorType_Internal:
      fmt::print("WGPU: Internal error: {}\n", message);
      break;

    case WGPUErrorType_Unknown:
      fmt::print("WGPU: Unknown error: {}\n", message);
      break;

    case WGPUErrorType_DeviceLost:
      fmt::print("WGPU: Device lost: {}\n", message);
      break;

    case WGPUErrorType_Force32:
      fmt::print("WGPU: Force32 error: {}\n", message);
      break;

    default:
      fmt::print("WGPU: Unknown error type: {}\n", message);
      break;
  }
}

void GPU::OnDeviceLost(WGPUDeviceLostReason reason, char const* message) {
  switch(reason) {
    case WGPUDeviceLostReason_Undefined:
      fmt::print("WGPU: Device lost (undefined), {}\n", message);
      break;
    case WGPUDeviceLostReason_Destroyed:
      fmt::print("WGPU: Device lost destroyed, {}\n", message);
      break;
    case WGPUDeviceLostReason_Force32:
      fmt::print("WGPU: Device lost force 32, {}\n", message);
      break;

    default:
      fmt::print("WGPU: Device lost (default), {}\n", message);
      break;
  }
}
