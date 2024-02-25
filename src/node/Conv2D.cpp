#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/NodePipeline.hpp"
#include "node/Conv2D.wgsl.hpp"

#include <cassert>
#include <iostream>

Node Conv2D(Node input, int kernel_size, int channels, int stride) {
  class Impl : public NodeImpl {
   public:
    std::string Name() override { return "Conv2D"; }

    std::vector<int> input_sizes_;
    std::vector<int> output_sizes_;
    int batch_size_ = 1;
    const int channels_ = 1;
    const int kernel_size_ = 1;
    const int stride_ = 1;

    Impl(Node input, int kernel_size, int channels, int stride)
        : NodeImpl(input),
          kernel_size_(kernel_size),
          channels_(channels),
          stride_(stride) {
      input_sizes_ = input->outputs[0].sizes();
      const int input_dx = input_sizes_[0];
      const int input_dy = input_sizes_[1];

      assert((input_dx - kernel_size_) % stride_ == 0);
      assert((input_dy - kernel_size_) % stride_ == 0);
      const int output_dx = (input_dx - kernel_size) / stride_ + 1;
      const int output_dy = (input_dy - kernel_size) / stride_ + 1;

      batch_size_ = input->outputs[0].TotalSize() / (input_dx * input_dy);
      output_sizes_ = {
          output_dx,
          output_dy,
          channels_,
          batch_size_,
      };

      outputs = {
          Tensor(output_sizes_),
      };
      outputs[0].SetName("outputs[0]");
      outputs[0].Fill(gpu(), 0.f);

      weights = {
          Tensor({kernel_size, kernel_size, channels_}),
      };
      const float noise = 0.1f / (kernel_size * kernel_size);
      weights[0].SetName("weights[0]");
      weights[0].FillRandomGaussian(gpu(), 0.f, noise);

      SetupGradients();

      wgpu::ShaderModule module = Shader(gpu(), fmt::format(wgsl::Conv2D,  //
                                                            input_dx,      //
                                                            input_dy,      //
                                                            channels_,     //
                                                            kernel_size,   //
                                                            stride,        //
                                                            batch_size_));

      pipeline_.Init(module, {
                                 &input->outputs[0],
                                 &input->outputs_gradients[0],
                                 &weights[0],
                                 &weights_gradients[0],
                                 &outputs[0],
                                 &outputs_gradients[0],
                             });
    }

    void Forward() override {
      pipeline_.Run("fn_output",                   //
                    (output_sizes_[0] + 15) / 16,  //
                    (output_sizes_[1] + 15) / 16,  //
                    channels_ * batch_size_        //
      );
    }

    void Backward() override {
      pipeline_.Run("fn_input_gradient",          //
                    (input_sizes_[0] + 15) / 16,  //
                    (input_sizes_[1] + 15) / 16,  //
                    batch_size_                   //
      );
      pipeline_.Run("fn_weight_gradient",      //
                    (kernel_size_ + 15) / 16,  //
                    (kernel_size_ + 15) / 16,  //
                    channels_                  //
      );
    }

    NodePipeline pipeline_{gpu()};
  };
  return std::make_shared<Impl>(input, kernel_size, channels, stride);
}
