#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/NodePipeline.hpp"
#include "node/Linear.wgsl.hpp"

#include <iostream>

Node Linear(Node input, int output_size) {
  class Impl : public NodeImpl {
   public:
    std::string Name() override { return "Linear"; }

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
      weights[0].FillRandomGaussian(gpu(), 0.f, 1.f / sqrtf(input_size_));
      weights[1].FillRandomGaussian(gpu(), 0.f, 1.f / sqrtf(input_size_));

      outputs = {
          Tensor({
              output_size_,
              batch_size_,
          }),
      };
      outputs[0].Fill(gpu(), 0.f);

      SetupGradients();

      wgpu::ShaderModule module = Shader(
          gpu(),
          fmt::format(wgsl::Linear, input_size_, output_size_, batch_size_));

      pipeline_.Init(module, {
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
      pipeline_.Run("fn_output",                           //
                    output_size_, (batch_size_ + 63) / 64  //
      );
    }
    void Backward() override {
      pipeline_.Run("fn_input_gradient",     //
                    input_size_,             //
                    (batch_size_ + 63) / 64  //
      );

      pipeline_.Run("fn_weights_gradient",  //
                    (input_size_ + 7) / 8,  //
                    (output_size_ + 7) / 8  //
      );

      pipeline_.Run("fn_bias_gradient",       //
                    (output_size_ + 63) / 64  //
      );
    }

    NodePipeline pipeline_{gpu()};
  };
  return std::make_shared<Impl>(input, output_size);
}
