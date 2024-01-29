#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/NodePipeline.hpp"

#include <iostream>

Node Sigmoid(Node input) {
  class Impl : public NodeImpl {
   public:
    std::string Name() override { return "Sigmoid"; }

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
          let x = input[id.x];
          output[id.x] = 1.0 / (1.0 + exp(-x));
        }}

        @compute @workgroup_size(256, 1, 1)
        fn fn_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {{
          if (id.x >= size) {{
            return;
          }}
          let x = input[id.x];
          let y = output[id.x];
          input_gradient[id.x] = y * (1.0 - y) * output_gradient[id.x];
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
