#ifndef NODE_HPP
#define NODE_HPP

#include <memory>
#include <unordered_set>
#include <vector>
#include "Tensor.hpp"
#include "node/NodePipeline.hpp"

class NodeImpl;
using Node = std::shared_ptr<NodeImpl>;
using NodePtr = NodeImpl*;

Node Input(GPU& gpu, std::vector<int> sizes);
Node Linear(Node input, int output_size);
Node Difference(Node a, Node b);
Node Squared(Node input);
Node Sigmoid(Node input);
Node Softmax(Node input);
Node CrossEntropy(Node a, Node b);
Node HuberLoss(Node input);
Node ReLU(Node input);
Node LeakyReLU(Node input);
Node Conv2D(Node input, int channels, int kernel_size, int stride = 1);
Node Conv2DTranspose(Node input, int channels, int kernel_size, int stride = 1);
Node MaxPool2D(Node input, int kernel_size);

class UpdateParams;

class NodeImpl {
 public:
  NodeImpl(GPU& gpu);
  NodeImpl(Node& input);
  NodeImpl(Node& input_a, Node& input_b);

  virtual ~NodeImpl() = default;

  std::vector<Tensor> weights;
  std::vector<Tensor> weights_gradients;

  std::vector<Tensor> outputs;
  std::vector<Tensor> outputs_gradients;

  virtual void Forward() {}
  virtual void Backward() {}
  virtual std::string Name() { return "Node"; }
  void UpdateParameters(float learning_rate);

  // A topology-sorted list of nodes to be used in the forward/backward pass.
  // The forward pass starts from the input node and ends at the output node.
  // The backward pass starts from the output node and ends at the input node.
  static std::vector<NodePtr> ForwardPassNodes(NodePtr input, NodePtr output);
  static std::vector<NodePtr> BackwardPassNodes(NodePtr input, NodePtr output);

  GPU& gpu() { return gpu_; }

 protected:
  void SetupGradients();

 private:
  void AddNode(Node& input);
  std::shared_ptr<UpdateParams> update_params_;

  std::vector<NodePipeline> pipeline_;

  static std::unordered_set<NodePtr> ForwardNodes(NodePtr input);
  static std::unordered_set<NodePtr> BackwardNodes(NodePtr input);
  static std::unordered_set<NodePtr> RelevantNodes(NodePtr input,
                                                   NodePtr output);

  GPU& gpu_;

  // Node connections:
  std::vector<Node> input_nodes;
  std::vector<NodePtr> output_nodes;
};

#endif
