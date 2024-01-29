#include <cassert>
#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/NodePipeline.hpp"

#include <iostream>

Node Difference(Node a, Node b) {
  class Impl : public NodeImpl { public:
    std::string Name() override { return "Difference"; }

    std::vector<int> sizes_;
    int size_ = 0;

    Impl(Node a, Node b) : NodeImpl(a, b) {
      assert(a->outputs[0].sizes() ==  //
             b->outputs[0].sizes());
      size_ = a->outputs[0].TotalSize();
      sizes_ = a->outputs[0].sizes();

      outputs = {Tensor({sizes_})};
      outputs[0].Fill(gpu(), 0.f);

      SetupGradients();

      wgpu::ShaderModule module = Shader(gpu(), fmt::format(R"(
        const size : u32 = {};

        // Input
        @group(0) @binding(0) var<storage, read_write> input_a: array<f32, size>;
        @group(0) @binding(1) var<storage, read_write> input_b: array<f32, size>;
        @group(0) @binding(2) var<storage, read_write> input_a_gradient: array<f32, size>;
        @group(0) @binding(3) var<storage, read_write> input_b_gradient: array<f32, size>;

        // Output
        @group(0) @binding(4) var<storage, read_write> output: array<f32, size>;
        @group(0) @binding(5) var<storage, read_write> output_gradient: array<f32, size>;

        @compute @workgroup_size(256, 1, 1)
        fn fn_output(@builtin(global_invocation_id) global_id: vec3<u32>) {{
          let x = global_id.x;
          if (x >= size) {{
            return;
          }}

          let a = input_a[x];
          let b = input_b[x];
          let diff = a - b;
          output[x] = diff;
        }}

        @compute @workgroup_size(256, 1, 1)
        fn fn_output_gradient(@builtin(global_invocation_id) global_id: vec3<u32>) {{
          let x = global_id.x;
          if (x >= size) {{
            return;
          }}

          let gradient = output_gradient[x];
          input_a_gradient[x] = gradient;
          input_b_gradient[x] = -gradient;
        }}
     )",
                                                            size_));

      pipeline_.Init(module, {
                                 &a->outputs[0],
                                 &b->outputs[0],
                                 &a->outputs_gradients[0],
                                 &b->outputs_gradients[0],
                                 &outputs[0],
                                 &outputs_gradients[0],
                             });
    }

    void Forward() override { pipeline_.Run("fn_output", (size_ + 255) / 256); }
    void Backward() override {
      pipeline_.Run("fn_output_gradient", (size_ + 255) / 256);
    }

    NodePipeline pipeline_{gpu()};
  };
  return std::make_shared<Impl>(a, b);
}
