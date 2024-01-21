#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include "GPU.hpp"

// An array of f16 values stored in the GPU. This is the input/output of a node.
class Tensor {
 public:
  Tensor() = default;
  Tensor(std::vector<int> size) : sizes_(size) {}

  // Write operations:
  void Write(GPU& gpu, std::vector<float> data);
  void Fill(GPU& gpu, float value);

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
  wgpu::Buffer buffer_;
};

#endif  // TENSOR_HPP
