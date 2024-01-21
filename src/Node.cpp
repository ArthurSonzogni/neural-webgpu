#include "Node.hpp"

#include <queue>
#include <unordered_set>
#include <vector>

NodeImpl::NodeImpl(GPU& gpu) : gpu_(gpu) {}

NodeImpl::NodeImpl(Node& input) : NodeImpl(input->gpu()) {
  input_nodes.push_back(input);
  input->output_nodes.push_back(this);
}

// Returns a topology-sorted list of nodes to be used in the forward pass,
// starting from the input node and ending at the output node.
std::vector<Node> NodeImpl::ForwardPassNodes(Node input, Node output) {
  std::unordered_set<Node> relevant_nodes = RelevantNodes(input, output);
  std::vector<Node> sorted_nodes;
  std::unordered_set<Node> visited;
  std::queue<Node> queue;
  queue.push(input);
  while (!queue.empty()) {
    Node node = queue.front();
    queue.pop();
    if (visited.count(node) == 0) {
      visited.insert(node);
      if (relevant_nodes.count(node) > 0) {
        sorted_nodes.push_back(node);
      }
      for (NodeImpl* output : node->output_nodes) {
        queue.push(Node(output));
      }
    }
  }
  return sorted_nodes;
}

// Returns a topology-sorted list of nodes to be used in the backward pass,
// starting from the output node and ending at the input node.
std::vector<Node> NodeImpl::BackwardPassNodes(Node input, Node output) {
  std::unordered_set<Node> relevant_nodes = RelevantNodes(input, output);
  std::vector<Node> sorted_nodes;
  std::unordered_set<Node> visited;
  std::queue<Node> queue;
  queue.push(output);
  while (!queue.empty()) {
    Node node = queue.front();
    queue.pop();
    if (visited.count(node) == 0) {
      visited.insert(node);
      if (relevant_nodes.count(node) > 0) {
        sorted_nodes.push_back(node);
      }
      for (Node input : node->input_nodes) {
        queue.push(input);
      }
    }
  }
  return sorted_nodes;
}

void NodeImpl::UpdateParameters() {

}

void NodeImpl::SetupGradients() {
  for (Tensor& parameter : weights) {
    weights_gradients.push_back(Tensor(parameter.sizes()));
    weights_gradients.back().Fill(gpu(), 0.f);
  }

  for (Tensor& output : outputs) {
    outputs_gradients.push_back(Tensor(output.sizes()));
    outputs_gradients.back().Fill(gpu(), 0.f);
  }
}

// Return the set of nodes that are reachable from the input node, and moving
// forward.
// static
std::unordered_set<Node> NodeImpl::ForwardNodes(Node input) {
  std::unordered_set<Node> visited;
  std::queue<Node> queue;
  queue.push(input);
  while (!queue.empty()) {
    Node node = queue.front();
    queue.pop();
    if (visited.count(node) == 0) {
      visited.insert(node);
      for (NodeImpl* output : node->output_nodes) {
        queue.push(Node(output));
      }
    }
  }
  return visited;
}

// static
std::unordered_set<Node> NodeImpl::BackwardNodes(Node output) {
  std::unordered_set<Node> visited;
  std::queue<Node> queue;
  queue.push(output);
  while (!queue.empty()) {
    Node node = queue.front();
    queue.pop();
    if (visited.count(node) == 0) {
      visited.insert(node);
      for (Node input : node->input_nodes) {
        queue.push(input);
      }
    }
  }
  return visited;
}

// static
std::unordered_set<Node> NodeImpl::RelevantNodes(Node input, Node output) {
  std::unordered_set<Node> forward_nodes = ForwardNodes(input);
  std::unordered_set<Node> backward_nodes = BackwardNodes(output);
  std::unordered_set<Node> relevant_nodes;
  for (Node node : forward_nodes) {
    if (backward_nodes.count(node) > 0) {
      relevant_nodes.insert(node);
    }
  }
  return relevant_nodes;
}
