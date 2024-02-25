#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/NodePipeline.hpp"
#include "node/MaxPool2D.wgsl.hpp"

#include <assert.hpp>
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
      ASSERT(input_sizes_.size() >= 2);
      ASSERT(input_sizes_[0] % kernel_size == 0);
      ASSERT(input_sizes_[1] % kernel_size == 0);

      output_sizes_ = input_sizes_;
      output_sizes_[0] /= kernel_size;
      output_sizes_[1] /= kernel_size;

      batch_size_ = input->outputs[0].TotalSize() / (input_sizes_[0] * input_sizes_[1]);

      outputs = {
          Tensor(output_sizes_),
      };
      outputs[0].Fill(gpu(), 0.f);

      SetupGradients();

      wgpu::ShaderModule module = Shader(gpu(), fmt::format(wgsl::MaxPool2D,  //
                                                            input_sizes_[0],  //
                                                            input_sizes_[1],  //
                                                            kernel_size,      //
                                                            batch_size_));    //

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
