#include <assert.hpp>
#include <iostream>

#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "node/Conv2D.wgsl.hpp"
#include "node/NodePipeline.hpp"

Node Conv2D(Node input, int kernel_size, int channels, int stride) {
  class Impl : public NodeImpl {
   public:
    std::string Name() override { return "Conv2D"; }

    std::vector<int> input_sizes_;
    std::vector<int> output_sizes_;
    int batch_size_ = 1;
    const int output_channels_ = 1;
    const int kernel_size_ = 1;
    const int stride_ = 1;

    Impl(Node input, int kernel_size, int channels, int stride)
        : NodeImpl(input),
          kernel_size_(kernel_size),
          output_channels_(channels),
          stride_(stride) {
      const int input_dimensions = input->outputs[0].sizes().size();
      ASSERT(input_dimensions, "Conv2D input must be 4D.");

      input_sizes_ = input->outputs[0].sizes();
      const int input_dx = input_sizes_[0];
      const int input_dy = input_sizes_[1];
      const int input_channels = input_sizes_[2];
      batch_size_ = input_sizes_[3];

      ASSERT((input_dx - kernel_size_) % stride_ == 0);
      ASSERT((input_dy - kernel_size_) % stride_ == 0);
      const int output_dx = (input_dx - kernel_size) / stride_ + 1;
      const int output_dy = (input_dy - kernel_size) / stride_ + 1;
      const int output_channels = channels;

      output_sizes_ = {
          output_dx,
          output_dy,
          output_channels,
          batch_size_,
      };

      outputs = {
          Tensor(output_sizes_),
      };
      outputs[0].SetName("outputs[0]");
      outputs[0].Fill(gpu(), 0.f);

      weights = {
          Tensor({
              kernel_size,
              kernel_size,
              input_channels,
              output_channels_,
          }),
      };
      const float noise = 0.1f / std::sqrt(          //
                                     kernel_size *   //
                                     kernel_size *   //
                                     input_channels  //
                                 );                  //
      weights[0].SetName("weights[0]");
      weights[0].FillRandomGaussian(gpu(), 0.f, noise);

      SetupGradients();

      wgpu::ShaderModule module =
          Shader(gpu(), fmt::format(wgsl::Conv2D,      //
                                    input_dx,          //
                                    input_dy,          //
                                    input_channels,    //
                                    output_channels_,  //
                                    kernel_size,       //
                                    stride,            //
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
                    (output_sizes_[0] + 7 / 8),  //
                    (output_sizes_[1] + 7 / 8),  //
                    (output_sizes_[2] *            //
                     output_sizes_[3])             //
      );
    }

    void Backward() override {
      pipeline_.Run("fn_input_gradient",          //
                    (input_sizes_[0] + 7) / 8,  //
                    (input_sizes_[1] + 7) / 8,  //
                    (input_sizes_[2] *            //
                     input_sizes_[3])             //
      );
      pipeline_.Run("fn_weight_gradient",               //
                    (weights[0].sizes()[0] + 7) / 8,  //
                    (weights[0].sizes()[1] + 7) / 8,  //
                    (weights[0].sizes()[2] *            //
                     weights[0].sizes()[3])             //
      );
    }

    NodePipeline pipeline_{gpu()};
  };
  return std::make_shared<Impl>(input, kernel_size, channels, stride);
}
