#ifndef MODEL_HPP
#define MODEL_HPP

#include <functional>
#include <span>
#include <vector>
#include "Node.hpp"

class Model {
 public:
   Model();
   Model& Input(Node input, std::function<std::span<float>(int)> generator);
   Model& Size(int size);
   Model& Minimize(Node output);
   Model& LearningRate(float learning_rate);
   Model& Epochs(int epochs);
   Model& BatchSize(int batch_size);

   void Execute();

 private:
  struct TrainInputArguments {
    Node node;
    std::function<std::span<float>(int)> generator;
  };

  std::vector<TrainInputArguments> inputs_;
  Node output_;
  float learning_rate_ = 0.01f;
  int epochs_ = 0;
  int size_ = 0;
};

#endif  // MODEL_HPP
