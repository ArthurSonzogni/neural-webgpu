#include "Tensor.hpp"
#include <cassert>
#include <random>
#include "fmt/format.h"

int Tensor::TotalSize() {
  int size = 1;
  for (int i : sizes_) {
    size *= i;
  }
  return size;
}

void Tensor::CreateBuffer(GPU& gpu) {
  if (buffer_) {
    return;
  }

  wgpu::BufferDescriptor bufferDesc = {
      .label = "Tensor",
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
               wgpu::BufferUsage::CopyDst,
      .size = TotalSize() * sizeof(float),
      .mappedAtCreation = false,
  };
  buffer_ = gpu.Device().CreateBuffer(&bufferDesc);
}

void Tensor::Fill(GPU& gpu, float value) {
  Write(gpu, std::vector<float>(TotalSize(), value));
}

void Tensor::FillRandomGaussian(GPU& gpu, float mean, float stddev) {
  static std::mt19937 rng;
  std::normal_distribution<float> random(mean, stddev);
  std::vector<float> values(TotalSize());
  for (auto& i : values) {
    i = random(rng);
  }
  Write(gpu, values);
}

void Tensor::Write(GPU& gpu, const std::vector<float>& data) {
  wgpu::Device& device = gpu.Device();
  CreateBuffer(gpu);
  assert(data.size() == TotalSize());
  gpu.Device().GetQueue().WriteBuffer(buffer_, 0, data.data(),
                                      data.size() * sizeof(float));
}

void Tensor::WritePartial(GPU& gpu, const std::span<float> data, int offset) {
  wgpu::Device& device = gpu.Device();
  CreateBuffer(gpu);
  assert(offset + data.size() <= TotalSize());
  gpu.Device().GetQueue().WriteBuffer(buffer_, offset * sizeof(float),
                                      data.data(), data.size() * sizeof(float));
}

void Tensor::WritePartialBatch(GPU& gpu,
                               const std::span<float> data,
                               int batch_offset) {
  assert(batch_offset < BatchSize());
  WritePartial(gpu, data, batch_offset * (TotalSize() / BatchSize()));
}

void Tensor::CopyTo(GPU& gpu, Tensor& other) {
  other.CopyFrom(gpu, *this);
}

void Tensor::CopyFrom(GPU& gpu, Tensor& other) {
  assert(sizes_ == other.sizes_);
  CreateBuffer(gpu);
  other.CreateBuffer(gpu);
  wgpu::CommandEncoder encoder = gpu.Device().CreateCommandEncoder();
  encoder.CopyBufferToBuffer(other.buffer_, 0, buffer_, 0,
                             TotalSize() * sizeof(float));
  wgpu::CommandBuffer commands = encoder.Finish();
  gpu.Device().GetQueue().Submit(1, &commands);
}

std::vector<float> Tensor::Read(GPU& gpu) {
  const int size = TotalSize();
  std::vector<float> out(size);

  gpu.Instance().ProcessEvents();
  wgpu::BufferDescriptor bufferDesc = {
      .label = "Readback buffer",
      .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
      .size = size * sizeof(float),
      .mappedAtCreation = false,
  };
  wgpu::Buffer map_buffer = gpu.Device().CreateBuffer(&bufferDesc);
  wgpu::CommandEncoder encoder = gpu.Device().CreateCommandEncoder();
  encoder.CopyBufferToBuffer(buffer_, 0, map_buffer, 0, size * sizeof(float));

  wgpu::CommandBuffer commands = encoder.Finish();
  gpu.Device().GetQueue().Submit(1, &commands);

  bool done = false;
  map_buffer.MapAsync(
      wgpu::MapMode::Read, 0, size * sizeof(float),
      [](WGPUBufferMapAsyncStatus status, void* userdata) {
        bool* done = reinterpret_cast<bool*>(userdata);
        *done = true;
      },
      reinterpret_cast<void*>(&done));

  while (!done) {
    gpu.Instance().ProcessEvents();
  }

  const float* output =
      (const float*)map_buffer.GetConstMappedRange(0, size * sizeof(float));
  for (int i = 0; i < size; i++) {
    out[i] = output[i];
  }
  map_buffer.Unmap();
  return out;
}
