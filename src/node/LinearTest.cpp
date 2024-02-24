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

TEST(Linear, Forward_Backward) {
  GPU gpu;

  const int batch_size = 2;
  const int input_size = 3;
  const int output_size = 3;

  Node input = Input(gpu, {input_size, batch_size});
  input->outputs[0].Write(gpu, {
                                   1, 2, 3,  // Batch 0
                                   4, 5, 6,  // Batch 1
                               });

  Node linear = Linear(input, output_size);
  linear->weights[0].Write(gpu, {
                                    1, 2, 3,  // Output 0
                                    4, 5, 6,  // Output 1
                                    7, 8, 9,  // Output 2
                                });
  linear->weights[1].Write(gpu, {
                                    1000,
                                    2000,
                                    3000,
                                });
  linear->Forward();
  const std::vector<float> expected_output = {
      1014, 2032, 3050,  // Batch 0
      1032, 2077, 3122,  // Batch 1
  };
  const std::vector<float> output = linear->outputs[0].Read(gpu);
  EXPECT_EQ(output, expected_output);

  linear->outputs_gradients[0].Write(gpu, {
                                              1014, 2032, 3050,  // Batch 0
                                              1032, 2077, 3122,  // Batch 1
                                          });

  const std::vector<float> expected_output_weights = {
      1014, 2032, 3050,  // Batch 0
      1032, 2077, 3122   // Batch 1
  };
  const std::vector<float> output_weights =
      linear->outputs_gradients[0].Read(gpu);
  EXPECT_EQ(output_weights, expected_output_weights);

  linear->Backward();

  const std::vector<float> expected_input_weights_gradients = {
      30492, 36588, 42684,  // Batch 0
      31194, 37425, 43656,  // Batch 1
  };
  const std::vector<float> input_weights =
      input->outputs_gradients[0].Read(gpu);
  EXPECT_EQ(input_weights, expected_input_weights_gradients);

  const std::vector<float> expected_weights_gradient = {
      5142,  7188,  9234,   // Output 0
      10340, 14449, 18558,  // Output 1
      15538, 21710, 27882,  // Output 2
  };
  const std::vector<float> weights_gradients =
      linear->weights_gradients[0].Read(gpu);
  EXPECT_EQ(weights_gradients, expected_weights_gradient);

  const std::vector<float> expected_bias_gradient = {
      2046, 4109, 6172,  // Output 0
  };
  const std::vector<float> bias_gradients =
      linear->weights_gradients[1].Read(gpu);
  EXPECT_EQ(bias_gradients, expected_bias_gradient);
}

TEST(Linear, Training) {
  GPU gpu;
  const int batch_size = 256;
  const int input_size = 2;
  const int output_size = 1;

  // Define the model:
  Node x = Input(gpu, {input_size, batch_size});
  Node y = Input(gpu, {output_size, batch_size});
  Node l = Linear(x, output_size);
  Node loss = HuberLoss(Difference(l, y));

  // Generate random training data. z = 3*x + 2*y + 1.
  static std::mt19937 rng;
  std::normal_distribution<float> random(0.0, 4);
  std::vector<std::vector<float>> input_data;
  std::vector<std::vector<float>> output_data;
  for (int i = 0; i < batch_size; ++i) {
    const float x = random(rng);
    const float y = random(rng);
    const float z = 3 * x + 2 * y + 1;
    input_data.push_back({x, y});
    output_data.push_back({z});
  }

  for (int i = 0; i < 1500; ++i) {
    Model()
        .Input(x, [&](int i) { return std::span(input_data[i]); })
        .Input(y, [&](int i) { return std::span(output_data[i]); })
        .Size(input_data.size())
        .Minimize(loss)
        .LearningRate(0.02f)
        .Epochs(1)
        .Execute();
  }

  // Test the model:
  std::vector<float> params_a = l->weights[0].Read(gpu);
  std::vector<float> params_b = l->weights[1].Read(gpu);
  EXPECT_NEAR(params_a.at(0), 3, 0.1);
  EXPECT_NEAR(params_a.at(1), 2, 0.1);
  EXPECT_NEAR(params_b.at(0), 1, 0.1);
}

TEST(Linear, MNIST) {
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
  xx = MaxPool2D(xx, 2);
  xx = Linear(xx, 30);
  xx = LeakyReLU(xx);
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
        .LearningRate(0.2f * std::pow(iteration + 1, -0.3f))
        .Epochs(1)
        .Execute();

    std::vector<std::vector<float>> predictions =
        Predict()
            .Input(x, [&](int i) { return std::span(test_examples[i].input); })
            .Output(predicted)
            .Size(test_examples.size())
            .Execute();

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
    }

    error /= predictions.size();

    fmt::print("Iteration: {} Error: {}\n", iteration, error);
    if (error < 0.05) {
      break;
    }
  }
}
