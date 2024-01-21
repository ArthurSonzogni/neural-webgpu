#ifndef NODE_HPP
#define NODE_HPP

#include <memory>
#include <unordered_set>
#include <vector>
#include "Tensor.hpp"

class NodeImpl;
using Node = std::shared_ptr<NodeImpl>;
using NodePtr = NodeImpl*;

Node Input(GPU& gpu, std::vector<int> sizes);
Node Linear(Node input, int output_size);

class NodeImpl {
 public:
  NodeImpl(GPU& gpu);
  NodeImpl(Node& input);

  virtual ~NodeImpl() = default;

  std::vector<Tensor> weights;
  std::vector<Tensor> weights_gradients;

  std::vector<Tensor> outputs;
  std::vector<Tensor> outputs_gradients;

  virtual void Forward() {}
  virtual void Backward() {}
  void UpdateParameters();

  // A topology-sorted list of nodes to be used in the forward/backward pass.
  // The forward pass starts from the input node and ends at the output node.
  // The backward pass starts from the output node and ends at the input node.
  static std::vector<Node> ForwardPassNodes(Node input, Node output);
  static std::vector<Node> BackwardPassNodes(Node input, Node output);

  GPU& gpu() { return gpu_; }

 protected:
  void SetupGradients();

 private:
  static std::unordered_set<Node> ForwardNodes(Node input);
  static std::unordered_set<Node> BackwardNodes(Node input);
  static std::unordered_set<Node> RelevantNodes(Node input, Node output);

  GPU& gpu_;

  // Node connections:
  std::vector<Node> input_nodes;
  std::vector<NodeImpl*> output_nodes;
};

#endif
