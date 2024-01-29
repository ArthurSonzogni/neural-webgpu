#ifndef NEURAL_WEBGPU_NODE_PIPELINE_HPP_
#define NEURAL_WEBGPU_NODE_PIPELINE_HPP_

#include <map>
#include <string>
#include <vector>
#include "GPU.hpp"
#include "Tensor.hpp"

// Helper class to run a compute shader on a node. It initializes the webgpu
// pipeline from a shader.
class NodePipeline {
 public:
  NodePipeline(GPU& gpu): gpu_(gpu) {}
  void Init(wgpu::ShaderModule module, std::vector<Tensor*> tensors);
  void Run(std::string entrypoint,
           int x_size = 1,
           int y_size = 1,
           int z_size = 1);

 private:
  GPU& gpu_;
  wgpu::ComputePipeline& GetPipeline(std::string entrypoint);
  wgpu::ShaderModule module_;
  wgpu::PipelineLayout pipeline_layout_;
  wgpu::BindGroup bindGroup_;
  std::map<std::string, wgpu::ComputePipeline> pipelines_;
};

#endif  // NEURAL_WEBGPU_NODE_PIPELINE_HPP_
