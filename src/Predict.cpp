#include "Predict.hpp"
#include <assert.hpp>
#include <iostream>
#include <fmt/format.h>

Predict::Predict() {}

Predict& Predict::Input(Node input,
                        std::function<std::span<float>(int)> generator) {
  inputs_.push_back({input, generator});
  return *this;
}

Predict& Predict::Output(Node output) {
  output_ = output;
  return *this;
}

Predict& Predict::Size(int size) {
  size_ = size;
  return *this;
}

std::vector<std::vector<float>> Predict::Execute() {
  std::vector<std::vector<float>> out;
  ASSERT(inputs_.size() > 0);

  NodePtr reference_node = inputs_[0].node.get();
  const int batch_size = reference_node->outputs[0].BatchSize();
  GPU& gpu = reference_node->gpu();

  std::vector<NodePtr> forward_nodes =
      NodeImpl::ForwardPassNodes(reference_node, output_.get());

  for (int g = 0; g < size_; g += batch_size) {
    // Fill inputs:
    for (PredictInputArgument& input : inputs_) {
      for (int i = 0; i < batch_size; ++i) {
        const int local_offset = (g + i) % size_;
        input.node->outputs[0].WritePartialBatch(
            gpu, input.generator(local_offset), i);
      }
    }

    // Forward pass:
    for (NodePtr node : forward_nodes) {
      node->Forward();
    }

    // Copy back the predicted output.
    std::vector<float> predictions = output_->outputs[0].Read(gpu);

    size_t begin = 0;
    for (int i = 0; i < batch_size; ++i) {
      if (out.size() >= size_) {
        break;
      }

      std::vector<float> prediction;
      size_t end = begin + predictions.size() / batch_size;
      std::copy(predictions.begin() + begin, predictions.begin() + end,
                std::back_inserter(prediction));
      out.push_back(prediction);
      begin = end;
    }
  }

  return out;
}
