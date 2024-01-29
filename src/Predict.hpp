#ifndef PREDICT_HPP
#define PREDICT_HPP

#include <functional>
#include <span>
#include <vector>
#include "Node.hpp"

// Usage:
// ------
//  std::vector<std::vector<float>> predictions =
//    Predict()
//      .Input(a, a_data)
//      .Input(b, b_data)
//      .Size(100)
//      .Execute();
//
class Predict {
 public:
  Predict();
  Predict& Input(Node input, std::function<std::span<float>(int)> generator);
  Predict& Output(Node output);
  Predict& Size(int size);

  std::vector<std::vector<float>> Execute();

 private:
  struct PredictInputArgument {
    Node node;
    std::function<std::span<float>(int)> generator;
  };

  std::vector<PredictInputArgument> inputs_;
  Node output_;
  size_t size_ = 0;
};

#endif  // PREDICT_HPP
