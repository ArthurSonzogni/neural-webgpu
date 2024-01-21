#ifndef SHADER_HPP
#define SHADER_HPP

#include "GPU.hpp"
#include <string>

wgpu::ShaderModule Shader(GPU& gpu, const std::string& code);

#endif  // SHADER_HPP
