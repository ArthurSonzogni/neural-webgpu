#include <assert.hpp>
#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/NodePipeline.hpp"
#include "node/Difference.wgsl.hpp"

#include <iostream>

Node Difference(Node a, Node b) {
  class Impl : public NodeImpl { public:
    std::string Name() override { return "Difference"; }

    std::vector<int> sizes_;
    int size_ = 0;

    Impl(Node a, Node b) : NodeImpl(a, b) {
      ASSERT(a->outputs[0].sizes() ==  //
             b->outputs[0].sizes());
      size_ = a->outputs[0].TotalSize();
      sizes_ = a->outputs[0].sizes();

      outputs = {Tensor({sizes_})};
      outputs[0].Fill(gpu(), 0.f);

      SetupGradients();

      wgpu::ShaderModule module =
          Shader(gpu(), fmt::format(wgsl::Difference, size_));

      pipeline_.Init(module, {
                                 &a->outputs[0],
                                 &b->outputs[0],
                                 &a->outputs_gradients[0],
                                 &b->outputs_gradients[0],
                                 &outputs[0],
                                 &outputs_gradients[0],
                             });
    }

    void Forward() override { pipeline_.Run("fn_output", (size_ + 63) / 64); }
    void Backward() override {
      pipeline_.Run("fn_output_gradient", (size_ + 63) / 64);
    }

    NodePipeline pipeline_{gpu()};
  };
  return std::make_shared<Impl>(a, b);
}
