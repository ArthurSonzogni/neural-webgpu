#include <iostream>
#include "Tensor.hpp"
#include "GPU.hpp"
#include "Shader.hpp"

int main() {
  GPU gpu;

  Tensor tensor({4});
  tensor.Write(gpu, {0.5f, 0.1f, 0.2f, 0.1f});

  // Create a shader module that applies a function to each element of the
  // buffer:
  wgpu::ShaderModule shader = Shader(gpu, R"(
    @group(0) @binding(0) var<storage, read_write> tensor: array<f32,4>;

    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        // Apply the function f to the buffer element at index id.x:
        tensor[id.x] *= sin(f32(id.x) * 123.f);
    }
  )");

  // Create the bind group layout:
  std::vector<wgpu::BindGroupLayoutEntry> bindGroupLayoutEntries{
      {
          .binding = 0,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer =
              {
                  .type = wgpu::BufferBindingType::Storage,
                  .hasDynamicOffset = false,
                  .minBindingSize = 4 * sizeof(float),
              },
      },
  };

  wgpu::BindGroupLayoutDescriptor bindGroupLayoutDescriptor{
      .label = "Bind group layout",
      .entryCount = bindGroupLayoutEntries.size(),
      .entries = bindGroupLayoutEntries.data(),
  };
  std::vector<wgpu::BindGroupLayout> bindGroupLayouts{
      gpu.Device().CreateBindGroupLayout(&bindGroupLayoutDescriptor),
  };

  // Create the pipeline layout:
  wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "Pipeline layout",
      .bindGroupLayoutCount = bindGroupLayouts.size(),
      .bindGroupLayouts = bindGroupLayouts.data(),
  };
  wgpu::PipelineLayout pipelineLayout =
      gpu.Device().CreatePipelineLayout(&pipelineLayoutDescriptor);

  // Create a compute pipeline that applies the shader module to the buffer:
  wgpu::ComputePipelineDescriptor computePipelineDesc = {
    .label = "Compute pipeline",
    .layout = pipelineLayout,
    .compute = {
      .module = shader,
      .entryPoint = "main",
    },
  };

  wgpu::ComputePipeline computePipeline =
      gpu.Device().CreateComputePipeline(&computePipelineDesc);

  // Create the bind group:
  std::vector<wgpu::BindGroupEntry> bindGroupEntries{
      {
          .binding = 0,
          .buffer = tensor.Buffer(),
          .offset = 0,
          .size = 4 * sizeof(float),
      },
  };
  wgpu::BindGroupDescriptor bindGroupDescriptor{
      .label = "Bind group",
      .layout = bindGroupLayouts[0],
      .entryCount = bindGroupEntries.size(),
      .entries = bindGroupEntries.data(),
  };
  wgpu::BindGroup bindGroup =
      gpu.Device().CreateBindGroup(&bindGroupDescriptor);

  // Send the command to the GPU:
  wgpu::CommandEncoder encoder = gpu.Device().CreateCommandEncoder();
  wgpu::ComputePassEncoder compute_pass = encoder.BeginComputePass();
  compute_pass.SetPipeline(computePipeline);
  compute_pass.SetBindGroup(0, bindGroup);
  compute_pass.DispatchWorkgroups(1);
  compute_pass.End();
  wgpu::CommandBuffer commands = encoder.Finish();
  gpu.Device().GetQueue().Submit(1, &commands);

  auto v = tensor.Read(gpu);

  std::cout << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << std::endl;


  return 0;
}
