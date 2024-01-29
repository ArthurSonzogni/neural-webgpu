#include "node/NodePipeline.hpp"
#include "fmt/format.h"

void NodePipeline::Init(wgpu::ShaderModule module,
                        std::vector<Tensor*> tensors) {
  module_ = module;

  std::vector<wgpu::BindGroupLayoutEntry> bindGroupLayoutEntries;
  for (uint32_t i = 0; i < tensors.size(); i++) {
    bindGroupLayoutEntries.push_back({
        .binding = i,
        .visibility = wgpu::ShaderStage::Compute,
        .buffer =
            {
                .type = wgpu::BufferBindingType::Storage,
                .minBindingSize = tensors[i]->TotalSize() * sizeof(float),
            },
    });
  }

  wgpu::BindGroupLayoutDescriptor bindGroupLayoutDescriptor{
      .label = "Bind group layout",
      .entryCount = bindGroupLayoutEntries.size(),
      .entries = bindGroupLayoutEntries.data(),
  };

  std::vector<wgpu::BindGroupLayout> bindGroupLayouts{
      gpu_.Device().CreateBindGroupLayout(&bindGroupLayoutDescriptor),
  };

  // Create the pipeline layout:
  wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "Pipeline layout",
      .bindGroupLayoutCount = bindGroupLayouts.size(),
      .bindGroupLayouts = bindGroupLayouts.data(),
  };
  pipeline_layout_ =
      gpu_.Device().CreatePipelineLayout(&pipelineLayoutDescriptor);

  // Create a compute pipeline that applies the shader module to the buffer:
  std::vector<wgpu::BindGroupEntry> bindGroupEntries;
  for (uint32_t i = 0; i < tensors.size(); i++) {
    bindGroupEntries.push_back({
        .binding = i,
        .buffer = tensors[i]->Buffer(),
        .size = tensors[i]->TotalSize() * sizeof(float),
    });
  };
  wgpu::BindGroupDescriptor bindGroupDescriptor{
      .label = "Bind group",
      .layout = bindGroupLayouts[0],
      .entryCount = bindGroupEntries.size(),
      .entries = bindGroupEntries.data(),
  };
  bindGroup_ = gpu_.Device().CreateBindGroup(&bindGroupDescriptor);
}

void NodePipeline::Run(std::string entrypoint,
                       int x_size,
                       int y_size,
                       int z_size) {
  wgpu::CommandEncoder encoder = gpu_.Device().CreateCommandEncoder();
  wgpu::ComputePassEncoder compute_pass = encoder.BeginComputePass();
  compute_pass.SetPipeline(GetPipeline(entrypoint));
  compute_pass.SetBindGroup(0, bindGroup_);
  compute_pass.DispatchWorkgroups(x_size, y_size, z_size);
  compute_pass.End();
  wgpu::CommandBuffer commands = encoder.Finish();
  gpu_.Device().GetQueue().Submit(1, &commands);
}

wgpu::ComputePipeline& NodePipeline::GetPipeline(std::string entrypoint) {
  if (pipelines_.count(entrypoint) == 0) {
    wgpu::ComputePipelineDescriptor description = {
        .label = "Compute pipeline",
        .layout = pipeline_layout_,
        .compute =
            {
                .module = module_,
                .entryPoint = entrypoint.c_str(),
            },
    };

    pipelines_[entrypoint] = gpu_.Device().CreateComputePipeline(&description);
  }
  return pipelines_[entrypoint];
}
