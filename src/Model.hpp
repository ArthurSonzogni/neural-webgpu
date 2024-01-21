#ifndef MODEL_HPP
#define MODEL_HPP

#include <span>
#include "Node.hpp"

class Model {
 public:
  Model(Node input, Node output) : input_(input), output_(output) {}

  void Train();
  void Predict(Tensor& input);

 private:
  Node input_;
  Node output_;
};

#endif  // MODEL_HPP
