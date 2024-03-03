#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/NodePipeline.hpp"
#include "node/Softmax.wgsl.hpp"

#include <iostream>

Node Softmax(Node input) {
  class Impl : public NodeImpl {
   public:
    std::string Name() override { return "Softmax"; }
    int size_;
    int batch_size_;
    std::vector<int> sizes_;

    Impl(Node input) : NodeImpl(input) {
      sizes_ = input->outputs[0].sizes();
      batch_size_ = input->outputs[0].BatchSize();
      size_ = input->outputs[0].TotalSize() / batch_size_;

      outputs = {Tensor({sizes_})};
      outputs[0].Fill(gpu(), 0.f);

      SetupGradients();

      wgpu::ShaderModule module =
          Shader(gpu(), fmt::format(wgsl::Softmax, size_, batch_size_));

      pipeline_.Init(module, {
                                 &input->outputs[0],
                                 &input->outputs_gradients[0],
                                 &outputs[0],
                                 &outputs_gradients[0],
                             });
    }

    void Forward() override {
      pipeline_.Run("fn_output", size_, (batch_size_ + 63) / 64);
    }
    void Backward() override {
      pipeline_.Run("fn_input_gradient", size_, (batch_size_ + 63) / 64);
    }

    NodePipeline pipeline_{gpu()};
  };
  return std::make_shared<Impl>(input);
}
