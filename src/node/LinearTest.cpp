#include <iostream>
#include "GPU.hpp"
#include "Model.hpp"
#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "gtest/gtest.h"

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
                                    1000, 2000, 3000,
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
      1014, 2032, 3050, // Batch 0
      1032, 2077, 3122 // Batch 1
  };
  const std::vector<float> output_weights =
      linear->outputs_gradients[0].Read(gpu);
  EXPECT_EQ(output_weights, expected_output_weights);

  linear->Backward();

  const std::vector<float> expected_input_weights_gradients = {
      30492, 36588, 42684, // Batch 0
      31194, 37425, 43656, // Batch 1
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

  const int batch_size = 100;
  const int input_size = 3;
  const int output_size = 3;
}
