#include <iostream>
#include "GPU.hpp"
#include "Model.hpp"
#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"
#include "gtest/gtest.h"

TEST(Squared, Forward_Backward) {
  GPU gpu;

  Node input = Input(gpu, {4, 2, 2});
  input->outputs[0].Write(gpu, {
      1,  2,  3,  4,   // Line 1
      5,  6,  7,  8,   // Line 2
      9,  10, 11, 12,  // Line 3
      13, 14, 15, 16,  // Line 4
  });

  Node squared = Squared(input);
  squared->Forward();
  const std::vector<float> expected_output = {
      1,   4,   9,   16,   // Line 1
      25,  36,  49,  64,   // Line 2
      81,  100, 121, 144,  // Line 3
      169, 196, 225, 256,  // Line 4
  };
  EXPECT_EQ(squared->outputs[0].Read(gpu), expected_output);

  squared->outputs_gradients[0].Write(gpu, {
      1,   4,   9,   16,   // Line 1
      25,  36,  49,  64,   // Line 2
      81,  100, 121, 144,  // Line 3
      169, 196, 225, 256,  // Line 4
  });
  squared->Backward();
  const std::vector<float> expected_input_gradient = {
      2,    16,   54,   128,   // Line 1
      250,  432,  686,  1024,  // Line 2
      1458, 2000, 2662, 3456,  // Line 3
      4394, 5488, 6750, 8192,  // Line 4
  };
  EXPECT_EQ(input->outputs_gradients[0].Read(gpu), expected_input_gradient);
}
