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

TEST(Conv2D, Forward_Backward) {
  GPU gpu;

  const int batch_size = 2;
  const int input_size = 3;

  Node input = Input(gpu, {input_size, input_size, batch_size});
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

  Node convolution = Conv2D(input, 2, 2);

  convolution->weights[0].Write(gpu, {
                                    // Channel 1
                                    1, 1,  //
                                    1, 1,  //

                                    // Channel 2,
                                    1, -1,  //
                                    1, -1,  //
                                });

  convolution->Forward();
  const std::vector<float> expected_output = {
      // Channel 1 (Batch 1)
      1, 1,  //
      1, 2,  //
      // Channel 2 (Batch 1)
      1, -1,  //
      -1, 0,  //
      // Channel 1 (Batch 2)
      1, 2,  //
      2, 1,  //
      // Channel 2 (Batch 2)
      -1, 0,  //
      0, 1,  //
  };
  const std::vector<float> output = convolution->outputs[0].Read(gpu);
  EXPECT_EQ(output, expected_output);
}

TEST(Conv2D, MNIST) {
  // Load the MNIST dataset:
  auto mnist = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(
      MNIST_DATA_LOCATION);
  std::vector<Example> training_examples =
      GetExamplesCentered(mnist.training_images, mnist.training_labels);
  std::vector<Example> test_examples =
      GetExamplesCentered(mnist.test_images, mnist.test_labels);

  GPU gpu;
  const int batch_size = 512;
  const int input_size = 28;
  const int output_size = 10;

  // Define the model:
  Node x = Input(gpu, {input_size, input_size, batch_size});
  Node y = Input(gpu, {output_size, batch_size});
  Node xx = x;

  // 28x28x512 -> 13x13x8x512
  xx = Conv2D(xx,               //
              /*kernel=*/6,     //
              /*channels=*/16,  //
              /*stride=*/2      //
  );
  xx = LeakyReLU(xx);

  // 2x2x32x512 -> 30x512
  xx = Linear(xx, 30);
  xx = LeakyReLU(xx);

  // 30x512 -> 10x512
  xx = Linear(xx, output_size);
  xx = Softmax(xx);

  Node predicted = xx;
  Node loss = CrossEntropy(y, predicted);

  for(int iteration = 0;; ++iteration) {
    Model()
        .Input(x, [&](int i) { return std::span(training_examples[i].input); })
        .Input(y, [&](int i) { return std::span(training_examples[i].output); })
        .Size(training_examples.size())
        .Minimize(loss)
        .LearningRate(0.3f * std::pow(iteration + 1, -0.3f))
        .Epochs(1)
        .Execute();
    fmt::print("Iteration: {}\n", iteration);

    std::vector<std::vector<float>> predictions =
        Predict()
            .Input(x, [&](int i) { return std::span(test_examples[i].input); })
            .Output(predicted)
            .Size(test_examples.size())
            .Execute();

    bool show = false;

    float error = 0;
    for(int i = 0; i < predictions.size(); ++i) {

      auto argmax = [](const std::vector<float>& v) {
        int argmax = 0;
        for(int i = 0; i < v.size(); ++i) {
          if(v[i] > v[argmax]) {
            argmax = i;
          }
        }
        return argmax;
      };

      int prediction_argmax = argmax(predictions[i]);
      int expected_argmax = argmax(test_examples[i].output);

      error += (prediction_argmax != expected_argmax);

      if (prediction_argmax == expected_argmax) {
        continue;
      }
      if (show) {
        continue;
      }
      show = true;

      fmt::print("Error: {} vs {}\n", prediction_argmax, expected_argmax);
      for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
          const auto value = test_examples[i].input[x + 28*(y)];

          if (value < -0.5) {
            fmt::print(" ");
            continue;
          }

          if (value < -0) {
            fmt::print(".");
            continue;
          }

          if (value < +0.5) {
            fmt::print("x");
            continue;
          }

          fmt::print("0");
        }
        fmt::print("\n");
      }
    }

    fmt::print("Iteration: {} Error: {}/{} = {}\n", iteration, error,
               predictions.size(), error / predictions.size());
    //if (error < 0.05) {
      //break;
    //}
  }
}
