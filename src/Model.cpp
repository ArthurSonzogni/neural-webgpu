#include "Model.hpp"
#include "fmt/format.h"

Model::Model() = default;

Model& Model::Input(Node input,
                    std::function<std::span<float>(int)> generator) {
  inputs_.push_back({input, generator});
  return *this;
}

Model& Model::Size(int size) {
  size_ = size;
  return *this;
}

Model& Model::Minimize(Node output) {
  output_ = output;
  return *this;
}

Model& Model::LearningRate(float learning_rate) {
  learning_rate_ = learning_rate;
  return *this;
}

Model& Model::Epochs(int epochs) {
  epochs_ = epochs;
  return *this;
}

void Model::Execute() {
  NodePtr reference_node = inputs_[0].node.get();
  const int batch_size = reference_node->outputs[0].BatchSize();
  GPU& gpu = reference_node->gpu();

  std::vector<NodePtr> backward_nodes =
      NodeImpl::BackwardPassNodes(reference_node, output_.get());
  std::vector<NodePtr> forward_nodes =
      NodeImpl::ForwardPassNodes(reference_node, output_.get());

  for (int g = 0; g < epochs_ * size_; g += batch_size) {
    // Fill inputs:
    for (TrainInputArguments& input : inputs_) {
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

    // We want to minimize the loss.
    output_->outputs[0].CopyTo(gpu, output_->outputs_gradients[0]);

    //std::vector<float> loss = output_->outputs[0].Read(gpu);
    //float sum = 0;
    //for (float l : loss) {
      //sum += l * l;
    //}
    //sum = sqrt(sum / loss.size());
    //fmt::print("Loss: {}\n", sum);

    // Backward pass:
    for (NodePtr node : backward_nodes) {
      node->Backward();
      gpu.Instance().ProcessEvents();
    }

    // Update parameters:
    for (NodePtr node : backward_nodes) {
      node->UpdateParameters(learning_rate_ / batch_size);
    }
  }
}
