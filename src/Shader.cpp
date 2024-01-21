#include "Shader.hpp"
#include "GPU.hpp"

wgpu::ShaderModule Shader(GPU& gpu, const std::string& code) {
  // Create a shader module that applies a function to each element of the
  // buffer:
  wgpu::ShaderModuleWGSLDescriptor wgslDesc;
  wgslDesc.code = code.c_str();
  wgpu::ShaderModuleDescriptor shaderModuleDescriptor{
      .nextInChain = &wgslDesc,
  };
  return gpu.Device().CreateShaderModule(&shaderModuleDescriptor);
}
