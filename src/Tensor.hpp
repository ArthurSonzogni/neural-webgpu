#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <span>
#include <vector>
#include "GPU.hpp"

// An array of f16 values stored in the GPU. This is the input/output of a node.
class Tensor {
 public:
  Tensor() = default;
  Tensor(std::vector<int> size) : sizes_(size) {}

  // Tensor is copyable. Both the copy and the original will point to the same
  // GPU buffer.
  Tensor(const Tensor& other);
  Tensor& operator=(const Tensor& other);

  // Name operations:
  void SetName(std::string name) { name_ = name; }

  // Write operations:
  void Write(GPU& gpu, const std::vector<float>& data);
  void WritePartial(GPU& gpu, const std::span<float> data, int offset);
  void WritePartialBatch(GPU& gpu, const std::span<float> data, int batch_offset);
  void Fill(GPU& gpu, float value);
  void FillRandomGaussian(GPU& gpu, float mean, float stddev);

  // Copy operations
  void CopyTo(GPU& gpu, Tensor& other);
  void CopyFrom(GPU& gpu, Tensor& other);

  // Read operations:
  std::vector<float> Read(GPU& gpu);

  wgpu::Buffer& Buffer() { return buffer_; }

  int TotalSize();
  int BatchSize() const { return sizes_.back(); }
  const std::vector<int>& sizes() const { return sizes_; }

 private:
  void CreateBuffer(GPU& gpu);

  std::vector<int> sizes_;
  std::string name_ = "Tensor";
  wgpu::Buffer buffer_;
};

#endif  // TENSOR_HPP
