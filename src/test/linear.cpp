#include <iostream>
#include "GPU.hpp"
#include "Model.hpp"
#include "Node.hpp"
#include "Shader.hpp"
#include "Tensor.hpp"

TEST(Linear, Forward) {
  GPU gpu;

  int batch_size = 1000;
  int mnist_width = 28;
  int mnist_height = 28;

  Node input = Input(gpu, {
                              mnist_width,
                              mnist_height,
                              batch_size,
                          });

  auto layers = [](Node X) {
    // X                // Size: 28x28x1000
    X = Linear(X, 10);  // Size: 10x1000
    return X;
  };

  Node output = layers(input);

  Model model(input, output);
  model.Train();
  model.Train();
  model.Train();
}

TEST(Linear, Forward) {
  GPU gpu;

  int batch_size = 1000;
  int mnist_width = 28;
  int mnist_height = 28;

  Node input = Input(gpu, {
                              mnist_width,
                              mnist_height,
                              batch_size,
                          });

  auto layers = [](Node X) {
    // X                // Size: 28x28x1000
    X = Linear(X, 10);  // Size: 10x1000
    return X;
  };

  Node output = layers(input);

  Model model(input, output);
  model.Train();
}
