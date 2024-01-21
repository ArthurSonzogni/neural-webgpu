#ifndef NEURAL_WEBGPU_NODE_PIPELINE_HPP_
#define NEURAL_WEBGPU_NODE_PIPELINE_HPP_

#include <map>
#include <string>
#include <vector>
#include "GPU.hpp"
#include "Tensor.hpp"

class NodePipeline {
 public:
  void Init(GPU& gpu, wgpu::ShaderModule module, std::vector<Tensor*> tensors);
  void Run(GPU& gpu,
           std::string entrypoint,
           int x_size,
           int y_size,
           int z_size);

 private:
  wgpu::ComputePipeline& GetPipeline(GPU& gpu, std::string entrypoint);
  wgpu::ShaderModule module_;
  wgpu::PipelineLayout pipeline_layout_;
  wgpu::BindGroup bindGroup_;
  std::map<std::string, wgpu::ComputePipeline> pipelines_;
};

#endif  // NEURAL_WEBGPU_NODE_PIPELINE_HPP_
