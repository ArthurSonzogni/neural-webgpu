#include <assert.hpp>
#include <iostream>
#include <random>
#include "Example.hpp"
#include "GPU.hpp"
#include "Model.hpp"
#include "Node.hpp"
#include "Predict.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "fmt/format.h"
#include "gtest/gtest.h"
#include "mnist/mnist_reader.hpp"

namespace {
std::vector<Example> GetExamplesCentered(
    const std::vector<std::vector<float>>& input,
    const std::vector<uint8_t>& output) {
  std::vector<Example> examples;
  for (size_t i = 0; i < input.size(); ++i) {
    std::vector<float> input_example = input[i];
    for (auto& p : input_example) {
      p /= 256.0f;
      p = 2.0 * p - 1;
    }

    std::vector<float> output_example(10, 0.f);
    output_example[output[i]] = 1.f;

    examples.push_back({
        input_example,
        output_example,
    });
  }

  return examples;
}
}  // namespace

TEST(Conv2DTranspose, Forward_Backward_stride_1) {
  GPU gpu;

  const int input_size = 3;
  const int batch_size = 2;

  Node input = Input(gpu, {input_size, input_size, 1, batch_size});
  input->outputs[0].Write(gpu, {
                                   // Batch 1
                                   1, 0, 0,  //
                                   0, 0, 1,  //
                                   0, 1, 0,  //

                                   // Batch 2
                                   0, 0, 1,  //
                                   0, 1, 0,  //
                                   1, 0, 0,  //
                               });

  Node convolution = Conv2DTranspose(input,
                                     /*kernel=*/2,
                                     /*channels=*/2,
                                     /*stride=*/1);

  convolution->weights[0].Write(gpu, {
                                         // Channel 1
                                         1, 1,  //
                                         1, 1,  //
                                         // Channel 2,
                                         1, -1, //
                                         1, -1, //
                                     });

  convolution->Forward();
  const std::vector<float> expected_output = {
      // Channel 1 (Batch 1)
      1, 1, 0, 0,  //
      1, 1, 1, 1,  //
      0, 1, 2, 1,  //
      0, 1, 1, 0,  //
      // Channel 2 (Batch 1)
      1, -1, 0, 0,   //
      1, -1, 1, -1,  //
      0, 1, 0, -1,   //
      0, 1, -1, 0,    //
      // Channel 1 (Batch 2)
      0, 0, 1, 1,  // 
      0, 1, 2, 1,  //
      1, 2, 1, 0,  //
      1, 1, 0, 0,  //
      // Channel 1 (Batch 2)
      0, 0, 1, -1,  // 
      0, 1, 0, -1,  //
      1, 0, -1, 0,  //
      1, -1, 0, 0,  //
  };
  const std::vector<float> output = convolution->outputs[0].Read(gpu);

  ASSERT(output == expected_output);
  EXPECT_EQ(output, expected_output);

  convolution->outputs_gradients[0].Write(
      gpu, {
               // Channel 1 (Batch 1)
               0, 0, 0, 0,  //
               0, 0, 0, 0,  //
               0, 0, 0, 0,  //
               0, 0, 0, 0,  //
                                     // Channel 2 (Batch 1)
               0, 0, 0, 0,  //
               0, 0, 0, 0,  //
               0, 0, 0, 0,  //
               0, 0, 0, 0,  //
                                     // Channel 1 (Batch 2)
               0, 0, 0, 0,  //
               0, 0, 0, 0,  //
               0, 1, 0, 0,  //
               0, 0, 0, 0,  //
                                     // Channel 2 (Batch 2)
               0, 0, 0, 0,  //
               0, 0, 0, 0,  //
               0, 0, 0, 0,  //
               0, 0, 0, 0,  //
           });

  convolution->Backward();
  const std::vector<float> expected_input_gradients = {
      // Batch 1
      0, 0, 0,  //
      0, 0, 0,  //
      0, 0, 0,  //

      // Batch 2
      0, 0, 0,  //
      1, 1, 0,  //
      1, 1, 0,  //
  };
  const std::vector<float> input_gradients =
      input->outputs_gradients[0].Read(gpu);
  EXPECT_EQ(input_gradients, expected_input_gradients);

  const std::vector<float> expected_weights_gradients = {
      // Channel 1
      0, 1,  //
      1, 0,  //

      // Channel 1
      0, 0,  //
      0, 0,  //
  };
  const std::vector<float> weights_gradients =
      convolution->weights_gradients[0].Read(gpu);
  EXPECT_EQ(weights_gradients, expected_weights_gradients);
}

TEST(Conv2DTranspose, Forward_Backward) {
  GPU gpu;

  const int input_size = 3;
  const int batch_size = 2;

  Node input = Input(gpu, {input_size, input_size, 1, batch_size});
  input->outputs[0].Write(gpu, {
                                   // Batch 1
                                   1, 0, 0,  //
                                   0, 0, 1,  //
                                   0, 1, 0,  //

                                   // Batch 2
                                   0, 0, 1,  //
                                   0, 1, 0,  //
                                   1, 0, 0,  //
                               });

  Node convolution = Conv2DTranspose(input,
                                     /*kernel=*/3,
                                     /*channels=*/2,
                                     /*stride=*/2);

  convolution->weights[0].Write(gpu, {
                                         // Channel 1
                                         1, 1, 1,  //
                                         1, 1, 1,  //
                                         1, 1, 1,  //
                                                   //
                                         // Channel 2,
                                         1, -1, 0,  //
                                         1, -1, 0,  //
                                         0, 0, 0,   //
                                     });

  convolution->Forward();
  const std::vector<float> expected_output = {
      // Channel 1 (Batch 1)
      1,1,1,0,0,0,0,//
      1,1,1,0,0,0,0,//
      1,1,1,0,1,1,1,//
      0,0,0,0,1,1,1,//
      0,0,1,1,2,1,1,//
      0,0,1,1,1,0,0,//
      0,0,1,1,1,0,0,//
      //
      // Channel 2 (Batch 1)
      1,-1,0,0 ,0,0 ,0,//
      1,-1,0,0 ,0,0 ,0,//
      0,0 ,0,0 ,1,-1,0,//
      0,0 ,0,0 ,1,-1,0,//
      0,0 ,1,-1,0,0 ,0,//
      0,0 ,1,-1,0,0 ,0,//
      0,0 ,0,0 ,0,0 ,0,//
      //
      // Channel 1 (Batch 2)
      0,0,0,0,1,1,1,//
      0,0,0,0,1,1,1,//
      0,0,1,1,2,1,1,//
      0,0,1,1,1,0,0,//
      1,1,2,1,1,0,0,//
      1,1,1,0,0,0,0,//
      1,1,1,0,0,0,0,//
      //
      // Channel 2 (Batch 2)
      0,0 ,0,0 ,1,-1,0,//
      0,0 ,0,0 ,1,-1,0,//
      0,0 ,1,-1,0,0 ,0,//
      0,0 ,1,-1,0,0 ,0,//
      1,-1,0,0 ,0,0 ,0,//
      1,-1,0,0 ,0,0 ,0,//
      0,0 ,0,0 ,0,0 ,0,//
  };
  const std::vector<float> output = convolution->outputs[0].Read(gpu);

  EXPECT_EQ(output, expected_output);

  convolution->outputs_gradients[0].Write(
      gpu, {
               // Channel 1 (Batch 1)
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 1, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
                                     // Channel 2 (Batch 1)
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
                                     // Channel 1 (Batch 2)
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
                                     // Channel 2 (Batch 2)
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
               0, 0, 0, 0, 0, 0, 0,  //
           });

  convolution->Backward();
  const std::vector<float> expected_input_gradients = {
      // Batch 1
      0, 0, 0,  //
      0, 1, 1,  //
      0, 1, 1,  //

      // Batch 2
      0, 0, 0,  //
      0, 0, 0,  //
      0, 0, 0,  //
  };
  const std::vector<float> input_gradients =
      input->outputs_gradients[0].Read(gpu);
  EXPECT_EQ(input_gradients, expected_input_gradients);

  const std::vector<float> expected_weights_gradients = {
      0, 0, 1,  //
      0, 0, 0,   //
      1, 0, 0,  //
                 //
      // Channel 2,
      0, 0, 0,  //
      0, 0, 0,  //
      0, 0, 0,  //
  };
  const std::vector<float> weights_gradients =
      convolution->weights_gradients[0].Read(gpu);
  EXPECT_EQ(weights_gradients, expected_weights_gradients);
}

TEST(Conv2DTranspose, MNIST) {
  // Load the MNIST dataset:
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_examples =
      GetExamplesCentered(mnist.training_images, mnist.training_labels);
  std::vector<Example> test_examples =
      GetExamplesCentered(mnist.test_images, mnist.test_labels);

  GPU gpu;
  const int batch_size = 1024;
  const int input_size = 28;
  const int output_size = 10;

  // Define the model:
  Node x = Input(gpu, {input_size, input_size, 1, batch_size});
  Node xx = x;

  // 28x28x1x512 -> 11x11x6x512
  xx = Conv2D(xx,              //
              /*kernel=*/8,    //
              /*channels=*/6,  //
              /*stride=*/2     //
  );
  xx = BatchNormalization(xx);
  xx = LeakyReLU(xx);

  // 11x11x6x512 -> 5x5x16x512
  xx = Conv2D(xx,               //
              /*kernel=*/3,     //
              /*channels=*/16,  //
              /*stride=*/2      //
  );
  xx = LeakyReLU(xx);

  // 5x5x16x512 -> 2x2x16x512
  xx = Conv2D(xx,               //
              /*kernel=*/3,     //
              /*channels=*/32,  //
              /*stride=*/2      //
  );
  xx = LeakyReLU(xx);

  // Dense layer:
  xx = Linear(xx, {10});
  xx = Linear(xx, {2, 2, 16});
  xx = LeakyReLU(xx);

  // 2x2x16x512 -> 5x5x16x512
  xx = Conv2DTranspose(xx,               //
                       /*kernel=*/3,     //
                       /*channels=*/16,  //
                       /*stride=*/2      //
  );
  xx = LeakyReLU(xx);

  // 5x5x16x512 -> 11x11x6x512
  xx = Conv2DTranspose(xx,              //
                       /*kernel=*/3,    //
                       /*channels=*/6,  //
                       /*stride=*/2     //
  );
  xx = LeakyReLU(xx);

  // 11x11x6x512 -> 28x28x1x512
  xx = Conv2DTranspose(xx,              //
                       /*kernel=*/8,    //
                       /*channels=*/1,  //
                       /*stride=*/2     //
  );

  Node predicted = xx;
  Node loss = HuberLoss(Difference(xx, x));

  for (int iteration = 0;; ++iteration) {
    Model()
        .Input(x, [&](int i) { return std::span(training_examples[i].input); })
        .Size(training_examples.size())
        .Minimize(loss)
        .LearningRate(1.0f * std::pow(iteration + 1, -0.3f))
        .Epochs(1)
        .Execute();

    std::vector<std::vector<float>> predictions =
        Predict()
            .Input(x, [&](int i) { return std::span(test_examples[i].input); })
            .Output(loss)
            .Size(1)
            .Execute();

    float error = 0;
    for (auto& prediction : predictions) {
      float e = 0.0f;
      for (auto& v : prediction) {
        e += v;
      }
      error += e;
    }
    error /= predictions.size();
    error /= 28.0f * 28.0f;
    error = std::sqrt(error);

    // Print the first predictions:
    for (int y = 0; y<28; ++y) {
      for (int x = 0; x<28; ++x) {
        const float value =
            predictions[iteration % predictions.size()][x + y * 28];
        //fmt::print("{:1f} ", value);
        if (value > 0.8f) {
          fmt::print("X");
          continue;
        }

        if (value > 0.3f) {
          fmt::print(":");
          continue;
        }

        if (value > 0.01f) {
          fmt::print(".");
          continue;
        }

        fmt::print(" ");
      }
      fmt::print("\n");
    }

    // Print the convolution weights:
    //{
      //fmt::print("Weights: \n");
      //auto weights = conv_1->weights[0].Read(gpu);
      //for(int y = 0; y<2; ++y) {
        //for(int x = 0; x<2; ++x) {
          //fmt::print("{:1f} ", weights[y * 2 + x]);
        //}
        //fmt::print("\n");
      //}
    //}
    //{
      //fmt::print("Weights: \n");
      //auto weights = conv_2->weights[0].Read(gpu);
      //for(int y = 0; y<2; ++y) {
        //for(int x = 0; x<2; ++x) {
          //fmt::print("{:1f} ", weights[y * 2 + x]);
        //}
        //fmt::print("\n");
      //}
    //}
    //{
      //fmt::print("Input: \n");
      //auto input = x->outputs[0].Read(gpu);
      //auto output = xx->outputs[0].Read(gpu);
      //auto l= loss->outputs[0].Read(gpu);
      //for(int y = 0; y<28; ++y) {
        //for(int x = 0; x<28; ++x) {
          //fmt::print("{:.2f}/{:.2f} ", l[x + y * 28],
                     //output[x + y * 28] - input[x + y * 28]);
        //}
        //fmt::print("\n");
      //}
    //}

    // Draw a progression bar:
    fmt::print("Iteration: {} Error: {}\n", iteration, error);
  }
}
