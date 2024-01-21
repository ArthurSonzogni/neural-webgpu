#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/NodePipeline.hpp"

#include <iostream>

Node Linear(Node input, int output_size) {
  class Impl : public NodeImpl {
   public:
    int batch_size_;
    int input_size_;
    int output_size_;

    Impl(Node input, int output_size) : NodeImpl(input) {
      batch_size_ = input->outputs[0].BatchSize();
      input_size_ = input->outputs[0].TotalSize() / batch_size_;
      output_size_ = output_size;

      weights = {
          Tensor({input_size_, output_size_}),  // Weights.
          Tensor({output_size_}),               // Bias.
      };
      weights[0].Fill(gpu(), 0.f);
      weights[1].Fill(gpu(), 0.f);

      outputs = {
          Tensor({
              output_size_,
              batch_size_,
          }),
      };
      outputs[0].Fill(gpu(), 0.f);

      SetupGradients();

      wgpu::ShaderModule module =
          Shader(gpu(), fmt::format(R"(
        const x_size : u32 = {};
        const y_size : u32 = {};
        const batch_size : u32 = {};

        // Input
        @group(0) @binding(0) var<storage, read_write> input: array<f32, x_size * batch_size>;
        @group(0) @binding(1) var<storage, read_write> input_gradient: array<f32, x_size * batch_size>;

        // Weights
        @group(0) @binding(2) var<storage, read_write> weights: array<f32, x_size * y_size>;
        @group(0) @binding(3) var<storage, read_write> bias: array<f32, y_size>;
        @group(0) @binding(4) var<storage, read_write> weights_gradient: array<f32, x_size * y_size>;
        @group(0) @binding(5) var<storage, read_write> bias_gradient: array<f32, y_size>;

        // Output
        @group(0) @binding(6) var<storage, read_write> output: array<f32, y_size * batch_size>;
        @group(0) @binding(7) var<storage, read_write> output_gradient: array<f32, y_size * batch_size>;

        @compute @workgroup_size(32, 4, 1)
        fn fn_output(@builtin(global_invocation_id) id: vec3<u32>) {{
            let y = id.x;
            let batch = id.y;
            if (y >= y_size || batch >= batch_size) {{
                return;
            }}

            var x_index = 0 + x_size * batch;
            var y_index = y + y_size * batch;
            var w_index = 0 + x_size * y;
            var sum : f32 = 0.f;
            for (var i = 0u; i < x_size; i = i + 1) {{
                sum += input[x_index] * weights[w_index];
                x_index++;
                w_index++;
            }}

            output[y_index] = sum + bias[y];
        }}

        @compute @workgroup_size(32, 4, 1)
        fn fn_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {{
            let x = id.x;
            let batch = id.y;
            if (x >= x_size || batch >= batch_size) {{
                return;
            }}

            var x_index = x + x_size * batch;
            var w_index = x;
            var sum : f32 = 0.0;
            for (var i = 0u; i < y_size; i = i + 1) {{
                sum += output_gradient[i + y_size * batch] * weights[w_index];
                w_index += x_size;
            }}

            input_gradient[x_index] = sum;
        }}

        @compute @workgroup_size(16, 16, 1)
        fn fn_weights_gradient(@builtin(global_invocation_id) id: vec3<u32>) {{
          var x_index = id.x;
          var y_index = id.y;
          if (x_index >= x_size || y_index >= y_size) {{
            return;
          }}

          let w = x_index + x_size * y_index;

          var sum : f32 = 0.0;
          for(var batch = 0u; batch < batch_size; batch++) {{
            sum += input[x_index] * output_gradient[y_index];
            x_index += x_size; // Next batch.
            y_index += y_size; // Next batch.
          }}

          weights_gradient[w] = sum;
        }}

        @compute @workgroup_size(64, 1, 1)
        fn fn_bias_gradient(@builtin(global_invocation_id) id: vec3<u32>) {{
          let y = id.x;
          if (y >= y_size) {{
            return;
          }}


          var sum : f32 = 0.0;
          var y_index = y;
          for(var batch = 0u; batch < batch_size; batch++) {{
            sum += output_gradient[y_index];
            y_index += y_size; // Next batch.
          }}

          bias_gradient[y] = sum;
        }}
     )",
                                    input_size_, output_size_, batch_size_));

      pipeline_.Init(gpu(), module,
                     {
                         &input->outputs[0],
                         &input->outputs_gradients[0],
                         &weights[0],
                         &weights[1],
                         &weights_gradients[0],
                         &weights_gradients[1],
                         &outputs[0],
                         &outputs_gradients[0],
                     });
    }

    void Forward() override {
      pipeline_.Run(gpu(), "fn_output",        //
                    (output_size_ + 31) / 32,  //
                    (batch_size_ + 3) / 4,     //
                    1                          //
      );
    }
    void Backward() override {
      pipeline_.Run(gpu(), "fn_input_gradient",  //
                    (input_size_ + 31) / 32,     //
                    (batch_size_ + 3) / 4,       //
                    1                            //
      );

      pipeline_.Run(gpu(), "fn_weights_gradient",  //
                    (input_size_ + 15) / 16,       //
                    (output_size_ + 15) / 16,      //
                    1                              //
      );

      pipeline_.Run(gpu(), "fn_bias_gradient",  //
                    (output_size_ + 63) / 64,   //
                    1,                          //
                    1                           //
      );
    }

    NodePipeline pipeline_;
  };
  return std::make_shared<Impl>(input, output_size);
}
