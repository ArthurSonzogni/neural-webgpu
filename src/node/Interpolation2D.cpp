#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/NodePipeline.hpp"
#include "node/Interpolation2D.wgsl.hpp"

#include <assert.hpp>
#include <iostream>

Node Interpolation2D(Node input, int width, int height) {
  class Impl : public NodeImpl {
   public:
    std::string Name() override { return "Interpolation2D"; }

    std::vector<int> input_sizes_;
    std::vector<int> output_sizes_;
    int batch_size_ = 1;

    Impl(Node input, int width, int height) : NodeImpl(input) {
      input_sizes_ = input->outputs[0].sizes();
      output_sizes_ = input_sizes_;
      output_sizes_[0] = width;
      output_sizes_[1] = height;
      outputs = {
          Tensor(output_sizes_),
      };
      outputs[0].Fill(gpu(), 0.f);

      batch_size_ =
          input->outputs[0].TotalSize() / (input_sizes_[0] * input_sizes_[1]);

      SetupGradients();

      wgpu::ShaderModule module =
          Shader(gpu(), fmt::format(wgsl::Interpolation2D,  //
                                    input_sizes_[0],   //
                                    input_sizes_[1],   //
                                    batch_size_,       //
                                    output_sizes_[0],  //
                                    output_sizes_[1]   //
                                    ));

      pipeline_.Init(module, {
                                 &input->outputs[0],
                                 &input->outputs_gradients[0],
                                 &outputs[0],
                                 &outputs_gradients[0],
                             });
    }

    void Forward() override {
      pipeline_.Run("fn_output",                 //
                    (output_sizes_[0] + 7) / 8,  //
                    (output_sizes_[1] + 7) / 8,  //
                    batch_size_                  //
      );
    }

    void Backward() override {
      pipeline_.Run("fn_input_gradient",        //
                    (input_sizes_[0] + 7) / 8,  //
                    (input_sizes_[1] + 7) / 8,  //
                    batch_size_                 //
      );
    }

    NodePipeline pipeline_{gpu()};
  };
  return std::make_shared<Impl>(input, width, height);
}
