#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/NodePipeline.hpp"

#include <cassert>
#include <iostream>

Node Conv2D(Node input, int kernel_size, int channels, int stride) {
  class Impl : public NodeImpl {
   public:
    std::string Name() override { return "Conv2D"; }

    std::vector<int> input_sizes_;
    std::vector<int> output_sizes_;
    int batch_size_ = 1;
    const int channels_ = 1;
    const int kernel_size_ = 1;
    const int stride_ = 1;

    Impl(Node input, int kernel_size, int channels, int stride)
        : NodeImpl(input),
          kernel_size_(kernel_size),
          channels_(channels),
          stride_(stride) {
      input_sizes_ = input->outputs[0].sizes();
      const int input_dx = input_sizes_[0];
      const int input_dy = input_sizes_[1];

      assert(input_dx % kernel_size == 0);
      assert(input_dy % kernel_size == 0);
      const int output_dx = (input_dx - kernel_size) / stride_ + 1;
      const int output_dy = (input_dy - kernel_size) / stride_ + 1;

      batch_size_ = input->outputs[0].TotalSize() / (input_dx * input_dy);
      output_sizes_ = {
          output_dx,
          output_dy,
          channels_,
          batch_size_,
      };

      outputs = {
          Tensor(output_sizes_),
      };
      outputs[0].Fill(gpu(), 0.f);

      weights = {
          Tensor({kernel_size, kernel_size, channels_}),
          Tensor({channels_}),
      };
      const float noise = 0.1f / (kernel_size * kernel_size);
      weights[0].FillRandomGaussian(gpu(), 0.f, noise);
      weights[1].FillRandomGaussian(gpu(), 0.f, noise);

      SetupGradients();

      wgpu::ShaderModule module = Shader(gpu(), fmt::format(R"(
        const input_dx : u32 = {};
        const input_dy : u32 = {};
        const channels : u32 = {};
        const kernel_size : u32 = {};
        const stride : u32 = {};
        const batch_size : u32 = {};

        const output_dx = (input_dx - kernel_size) / stride + 1;
        const output_dy = (input_dy - kernel_size) / stride + 1;

        const input_size = input_dx * input_dy * batch_size;
        const output_size = output_dx * output_dy * channels * batch_size;

        // Input
        @group(0) @binding(0) var<storage, read_write> input: array<f32, input_size>;
        @group(0) @binding(1) var<storage, read_write> input_gradient: array<f32, input_size>;

        // Params
        @group(0) @binding(2) var<storage, read_write> weights: array<f32, kernel_size * kernel_size * channels>;
        @group(0) @binding(3) var<storage, read_write> weights_gradient: array<f32, kernel_size * kernel_size * channels>;

        // Output
        @group(0) @binding(4) var<storage, read_write> output: array<f32, output_size>;
        @group(0) @binding(5) var<storage, read_write> output_gradient: array<f32, output_size>;


        @compute @workgroup_size(16, 16, 1)
        fn fn_output(@builtin(global_invocation_id) id: vec3<u32>) {{
          let x = id.x;
          let y = id.y;
          let c = id.z % channels;
          let b = id.z / channels;

          if (x >= output_dx || y >= output_dy || b >= batch_size) {{
            return;
          }}

          var sum = 0.f;
          for (var i : u32 = 0; i < kernel_size; i++) {{
            for (var j : u32 = 0; j < kernel_size; j++) {{
              let input_index = (i + stride * x) + input_dx * (
                                (j + stride * y) + input_dy * (
                                b
              ));
              let weight_index = (i + kernel_size * (
                                 (j + kernel_size * (
                                 c
              ))));
              sum += input[input_index] * weights[weight_index];
            }}
          }}
          let output_index = x + output_dx * (
                             y + output_dy * (
                             c + channels * (
                             b 
          )));
          output[output_index] = sum;
        }}

        @compute @workgroup_size(16, 16, 1)
        fn fn_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {{
          let x = id.x;
          let y = id.y;
          let b = id.z;

          if (x >= input_dx || y >= input_dy) {{
            return;
          }}

          let output_x = x / stride;
          let output_y = y / stride;
          var sum = 0.0;
          for (var c : u32 = 0; c < channels; c++) {{
            for (var i : u32 = 0; i < kernel_size; i++) {{
              for (var j : u32 = 0; j < kernel_size; j++) {{
                let output_index = output_x + output_dx * (
                                   output_y + output_dy * (
                                   c + channels * (
                                   b
                )));
                let weight_index = i + kernel_size * (
                                   j + kernel_size * (
                                   c
                ));
                sum += output_gradient[output_index] * weights[weight_index];
              }}
            }}
          }}
          let input_index = x + input_dx * (
                            y + input_dy * (
                            b
          ));
          input_gradient[input_index] = sum;
        }}

        @compute @workgroup_size(16, 16, 1)
        fn fn_weight_gradient(@builtin(global_invocation_id) id: vec3<u32>) {{
          let x = id.x;
          let y = id.y;
          let c = id.z;

          if (x >= kernel_size || y >= kernel_size) {{
            return;
          }}
          
          var sum = 0.0;
          for (var b : u32 = 0; b < batch_size; b++) {{
            for (var i : u32 = 0; i < output_dx; i++) {{
              for (var j : u32 = 0; j < output_dy; j++) {{
                let output_index = i + output_dx * (
                                   j + output_dy * (
                                   c + channels * (
                                   b
                )));
                let input_index = (i + stride * x) + input_dx * (
                                  (j + stride * y) + input_dy * (
                                  b
                ));
                sum += output_gradient[output_index] * input[input_index];
              }}
            }}
          }}

          let weight_index = x + kernel_size * (
                             y + kernel_size * (
                             c
          ));
          weights_gradient[weight_index] = sum;
        }}
        )",
                                                            input_dx,     //
                                                            input_dy,     //
                                                            channels_,    //
                                                            kernel_size,  //
                                                            stride,       //
                                                            batch_size_));

      pipeline_.Init(module, {
                                 &input->outputs[0],
                                 &input->outputs_gradients[0],
                                 &weights[0],
                                 &weights_gradients[0],
                                 &outputs[0],
                                 &outputs_gradients[0],
                             });
    }

    void Forward() override {
      pipeline_.Run("fn_output",                   //
                    (output_sizes_[0] + 15) / 16,  //
                    (output_sizes_[1] + 15) / 16,  //
                    channels_ * batch_size_        //
      );
    }

    void Backward() override {
      pipeline_.Run("fn_input_gradient",          //
                    (input_sizes_[0] + 15) / 16,  //
                    (input_sizes_[1] + 15) / 16,  //
                    batch_size_                   //
      );
      pipeline_.Run("fn_weight_gradient",      //
                    (kernel_size_ + 15) / 16,  //
                    (kernel_size_ + 15) / 16,  //
                    channels_                  //
      );
    }

    NodePipeline pipeline_{gpu()};
  };
  return std::make_shared<Impl>(input, kernel_size, channels, stride);
}
