#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/NodePipeline.hpp"

#include <iostream>

Node LeakyReLU(Node input) {
  class Impl : public NodeImpl {
   public:
    std::string Name() override { return "LeakyReLU"; }

    int size_;
    std::vector<int> sizes_;

    Impl(Node input) : NodeImpl(input) {
      size_ = input->outputs[0].TotalSize();
      sizes_ = input->outputs[0].sizes();

      outputs = {Tensor({sizes_})};
      outputs[0].Fill(gpu(), 0.f);

      SetupGradients();

      wgpu::ShaderModule module = Shader(gpu(), fmt::format(R"(
        const size : u32 = {};

        // Input
        @group(0) @binding(0) var<storage, read_write> input: array<f32, size>;
        @group(0) @binding(1) var<storage, read_write> input_gradient: array<f32, size>;

        // Output
        @group(0) @binding(2) var<storage, read_write> output: array<f32, size>;
        @group(0) @binding(3) var<storage, read_write> output_gradient: array<f32, size>;

        @compute @workgroup_size(256, 1, 1)
        fn fn_output(@builtin(global_invocation_id) id: vec3<u32>) {{
          if (id.x >= size) {{
            return;
          }}

          let value = input[id.x];
          if (value > 0) {{
            output[id.x] = value;
          }} else {{
            output[id.x] = 0.01 * value;
          }}
        }}

        @compute @workgroup_size(256, 1, 1)
        fn fn_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {{
          if (id.x >= size) {{
            return;
          }}

          let value = input[id.x];
          if (value > 0) {{
            input_gradient[id.x] = output_gradient[id.x];
          }} else {{
            input_gradient[id.x] = 0.01 * output_gradient[id.x];
          }}
        }}

     )",
                                                            size_));

      pipeline_.Init(module, {
                                 &input->outputs[0],
                                 &input->outputs_gradients[0],
                                 &outputs[0],
                                 &outputs_gradients[0],
                             });
    }

    void Forward() override {
      pipeline_.Run("fn_output",          //
                    (size_ + 255) / 256,  //
                    1,                    //
                    1                     //
      );
    }
    void Backward() override {
      pipeline_.Run("fn_input_gradient",  //
                    (size_ + 255) / 256,  //
                    1,                    //
                    1                     //
      );
    }

    NodePipeline pipeline_{gpu()};
  };
  return std::make_shared<Impl>(input);
}
