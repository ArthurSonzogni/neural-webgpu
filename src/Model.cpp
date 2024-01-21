#include "Model.hpp"

void Model::Train() {
  std::vector<Node> forward_nodes = NodeImpl::ForwardPassNodes(input_, output_);
  std::vector<Node> backward_nodes =
      NodeImpl::BackwardPassNodes(input_, output_);

  // Forward pass:
  for (Node node : forward_nodes) {
    node->Forward();
  }

  // Backward pass:
  for (Node node : backward_nodes) {
    node->Backward();
  }

  // Update parameters:
  for (Node node : backward_nodes) {
    node->UpdateParameters();
  }
}

void Model::Predict(Tensor& input) {
  //
}
