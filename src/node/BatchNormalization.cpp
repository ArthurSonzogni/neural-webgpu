#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/NodePipeline.hpp"
#include "node/BatchNormalization.wgsl.hpp"

#include <iostream>

Node BatchNormalization(Node input) {
  class Impl : public NodeImpl {
   public:
    std::string Name() override { return "BatchNormalization"; }

    int size_;
    std::vector<int> sizes_;

    Impl(Node input) : NodeImpl(input) {
      size_ = input->outputs[0].TotalSize();
      sizes_ = input->outputs[0].sizes();

      outputs = {Tensor({sizes_})};
      outputs[0].Fill(gpu(), 0.f);

      weights = {
          Tensor({2}),
      };
      weights[0].Write(gpu(), {
                                  1.f,  // scale
                                  0.f,  // bias
                              });

      SetupGradients();

      wgpu::ShaderModule module =
          Shader(gpu(), fmt::format(wgsl::BatchNormalization, size_));

      pipeline_.Init(module, {
                                 &weights[0],
                                 &input->outputs[0],
                                 &input->outputs_gradients[0],
                                 &outputs[0],
                                 &outputs_gradients[0],
                             });
    }

    void Forward() override {
      pipeline_.Run("fn_compute_weight", 64);
      pipeline_.Run("fn_output",          //
                    (size_ + 63) / 64,  //
                    1,                    //
                    1                     //
      );
    }
    void Backward() override {
      pipeline_.Run("fn_input_gradient",  //
                    (size_ + 63) / 64,  //
                    1,                    //
                    1                     //
      );
    }

    NodePipeline pipeline_{gpu()};
  };
  return std::make_shared<Impl>(input);
}
