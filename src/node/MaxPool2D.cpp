#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/NodePipeline.hpp"

#include <cassert>
#include <iostream>

Node MaxPool2D(Node input, int kernel_size) {
  class Impl : public NodeImpl {
   public:
    std::string Name() override { return "MaxPool2D"; }

    std::vector<int> input_sizes_;
    std::vector<int> output_sizes_;
    int batch_size_ = 1;

    Impl(Node input, int kernel_size) : NodeImpl(input) {
      input_sizes_ = input->outputs[0].sizes();
      assert(input_sizes_.size() >= 2);
      assert(input_sizes_[0] % kernel_size == 0);
      assert(input_sizes_[1] % kernel_size == 0);

      output_sizes_ = input_sizes_;
      output_sizes_[0] /= kernel_size;
      output_sizes_[1] /= kernel_size;

      batch_size_ = input->outputs[0].TotalSize() / (input_sizes_[0] * input_sizes_[1]);

      outputs = {
          Tensor(output_sizes_),
      };
      outputs[0].Fill(gpu(), 0.f);

      SetupGradients();

      wgpu::ShaderModule module =
          Shader(gpu(), fmt::format(R"(
        const input_dx : u32 = {};
        const input_dy : u32 = {};
        const kernel_size : u32 = {};
        const output_dx : u32 = input_dx / kernel_size;
        const output_dy : u32 = input_dy / kernel_size;
        const batch_size : u32 = {};

        const input_size = input_dx * input_dy * batch_size;
        const output_size = output_dx * output_dy * batch_size;

        // Input
        @group(0) @binding(0) var<storage, read_write> input: array<f32, input_size>;
        @group(0) @binding(1) var<storage, read_write> input_gradient: array<f32, input_size>;

        // Output
        @group(0) @binding(2) var<storage, read_write> output: array<f32, output_size>;
        @group(0) @binding(3) var<storage, read_write> output_gradient: array<f32, output_size>;

        @compute @workgroup_size(16, 16, 1)
        fn fn_output(@builtin(global_invocation_id) id: vec3<u32>) {{
          let x = id.x;
          let y = id.y;
          let b = id.z;

          if (x >= output_dx || y >= output_dy) {{
            return;
          }}

          var max_value : f32 = -3.402823466e+38;
          for (var i : u32 = 0; i < kernel_size; i++) {{
            for (var j : u32 = 0; j < kernel_size; j++) {{
              let input_index = (i + kernel_size * x) + input_dx * (
                                (j + kernel_size * y) + input_dy * (
                                b
              ));
              max_value = max(max_value, input[input_index]);
            }}
          }}
          let output_index = x + output_dx * (y + output_dy * b);
          output[output_index] = max_value;
        }}

        @compute @workgroup_size(16, 16, 1)
        fn fn_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {{
          let x = id.x;
          let y = id.y;
          let b = id.z;

          if (x >= input_dx || y >= input_dy) {{
            return;
          }}

          let output_x = x / kernel_size;
          let output_y = y / kernel_size;

          let output_index = output_x + output_dx * (
                             output_y + output_dy * (
                             b
          )); 
          let output_gradient_value = output_gradient[output_index];

          for (var i : u32 = 0; i < kernel_size; i++) {{
            for (var j : u32 = 0; j < kernel_size; j++) {{
              let input_index = (i + kernel_size * output_x) + input_dx * (
                                (j + kernel_size * output_y) + input_dy * (
                                b
              ));
              let input_value = input[input_index];
              if (input_value == output[output_index]) {{
                input_gradient[input_index] = output_gradient_value;
              }}
            }}
          }} 
        }}
     )",
                                    input_sizes_[0],   //
                                    input_sizes_[1],   //
                                    kernel_size,       //
                                    batch_size_));

      pipeline_.Init(module, {
                                 &input->outputs[0],
                                 &input->outputs_gradients[0],
                                 &outputs[0],
                                 &outputs_gradients[0],
                             });
    }

    void Forward() override {
      pipeline_.Run("fn_output",                   //
                    (output_sizes_[0] + 15) / 16,  //
                    (output_sizes_[1] + 15) / 16,  //
                    batch_size_                    //
      );                                           //
    }
    void Backward() override {
      pipeline_.Run("fn_input_gradient",          //
                    (input_sizes_[0] + 15) / 16,  //
                    (input_sizes_[1] + 15) / 16,  //
                    batch_size_                   //
      );
    }

    NodePipeline pipeline_{gpu()};
  };
  return std::make_shared<Impl>(input, kernel_size);
}
