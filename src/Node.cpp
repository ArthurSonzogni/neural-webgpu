#include "Node.hpp"

#include <fmt/format.h>
#include <queue>
#include <unordered_set>
#include <vector>
#include "Shader.hpp"

class UpdateParams {
 public:
  static std::shared_ptr<UpdateParams> Get(GPU& gpu) {
    static std::map<GPU*, std::weak_ptr<UpdateParams>> instances;

    if (instances.count(&gpu) == 1 && !instances[&gpu].expired()) {
      return instances[&gpu].lock();
    }

    auto instance = std::make_shared<UpdateParams>(gpu);
    instances[&gpu] = instance;
    return instance;
  }

  UpdateParams(GPU& gpu) {
    learning_rate.Write(gpu, {0.01f});

    module = Shader(gpu, R"(
      @group(0) @binding(0) var<storage, read_write> learning_rate: f32;
      @group(0) @binding(1) var<storage, read_write> weights: array<f32>;
      @group(0) @binding(2) var<storage, read_write> weights_gradients: array<f32>;
      @group(0) @binding(3) var<storage, read_write> weights_gradients_squared_sum: array<f32>;
      @group(0) @binding(4) var<storage, read_write> weights_momentum: array<f32>;

      const epsilon: f32 = 1e-8;
      const beta_1 = 0.9;
      const beta_2 = 0.99;

      @compute @workgroup_size(256, 1, 1)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let x = global_id.x;
        if (x >= arrayLength(&weights)) {
          return;
        }

        let gradient = weights_gradients[x];
        weights_gradients_squared_sum[x] = mix(
          gradient * gradient,
          weights_gradients_squared_sum[x],
          beta_2
        );

        weights_momentum[x] = mix(
          gradient,
          weights_momentum[x],
          beta_1
        );

        weights[x] -= learning_rate * weights_momentum[x] / (
          sqrt(weights_gradients_squared_sum[x]) +
          epsilon
        );
      }
    )");
  }

  wgpu::ShaderModule module;
  Tensor learning_rate{{1}};
};

NodeImpl::NodeImpl(GPU& gpu) : gpu_(gpu) {}

NodeImpl::NodeImpl(Node& input) : NodeImpl(input->gpu()) {
  AddNode(input);
}

NodeImpl::NodeImpl(Node& input_a, Node& input_b): NodeImpl(input_a->gpu()) {
  AddNode(input_a);
  AddNode(input_b);
}

void NodeImpl::AddNode(Node& input) {
  input_nodes.push_back(input);
  input->output_nodes.push_back(this);
}


// Returns a topology-sorted list of nodes to be used in the forward pass,
// starting from the input node and ending at the output node.
std::vector<NodePtr> NodeImpl::ForwardPassNodes(NodePtr input, NodePtr output) {
  std::unordered_set<NodePtr> relevant_nodes = RelevantNodes(input, output);
  std::vector<NodePtr> toposort;
  std::unordered_set<NodePtr> visited;
  auto visit = std::function<void(NodePtr)>();
  visit = [&](NodePtr node) {
    if (visited.count(node)) {
      return;
    }
    visited.insert(node);
    if (!relevant_nodes.count(node)) {
      return;
    }
    for (Node input : node->input_nodes) {
      visit(input.get());
    }
    toposort.push_back(node);
  };
  visit(output);
  return toposort;
}

// Returns a topology-sorted list of nodes to be used in the backward pass,
// starting from the output node and ending at the input node.
std::vector<NodePtr> NodeImpl::BackwardPassNodes(NodePtr input,
                                                 NodePtr output) {
  std::unordered_set<NodePtr> relevant_nodes = RelevantNodes(input, output);
  std::vector<NodePtr> toposort;
  std::unordered_set<NodePtr> visited;
  auto visit = std::function<void(NodePtr)>();
  visit = [&](NodePtr node) {
    if (visited.count(node)) {
      return;
    }
    visited.insert(node);
    if (!relevant_nodes.count(node)) {
      return;
    }
    for (NodeImpl* output : node->output_nodes) {
      visit(NodePtr(output));
    }
    toposort.push_back(node);
  };
  visit(input);
  return toposort;
}

void NodeImpl::UpdateParameters(float learning_rate) {
  for (int i = 0; i < weights.size(); ++i) {
    update_params_->learning_rate.Write(gpu(), {learning_rate});
    pipeline_[i].Run("main", (weights[i].TotalSize() + 255) / 256);
  }
}

void NodeImpl::SetupGradients() {
  update_params_ = UpdateParams::Get(gpu());
  wgpu::ShaderModule module = update_params_->module;

  for (Tensor& parameter : weights) {
    weights_gradients.push_back(Tensor(parameter.sizes()));
    weights_gradients.back().Fill(gpu(), 0.f);

    weights_gradients_squared_sum.push_back(Tensor(parameter.sizes()));
    weights_gradients_squared_sum.back().Fill(gpu(), 1.f);

    weights_momentum.push_back(Tensor(parameter.sizes()));
    weights_momentum.back().Fill(gpu(), 0.f);
  }

  for (int i = 0; i < weights.size(); ++i) {
    pipeline_.emplace_back(gpu());
    pipeline_.back().Init(module, {
                                      &update_params_->learning_rate,
                                      &weights[i],
                                      &weights_gradients[i],
                                      &weights_gradients_squared_sum[i],
                                      &weights_momentum[i],
                                  });
  }

  for (Tensor& output : outputs) {
    outputs_gradients.push_back(Tensor(output.sizes()));
    outputs_gradients.back().Fill(gpu(), 0.f);
  }
}

// Return the set of nodes that are reachable from the input node, and moving
// forward.
// static
std::unordered_set<NodePtr> NodeImpl::ForwardNodes(NodePtr input) {
  std::unordered_set<NodePtr> visited;
  std::queue<NodePtr> queue;
  queue.push(input);
  while (!queue.empty()) {
    NodePtr node = queue.front();
    queue.pop();
    if (visited.count(node) == 0) {
      visited.insert(node);
      for (NodeImpl* output : node->output_nodes) {
        queue.push(NodePtr(output));
      }
    }
  }
  return visited;
}

// static
std::unordered_set<NodePtr> NodeImpl::BackwardNodes(NodePtr output) {
  std::unordered_set<NodePtr> visited;
  std::queue<NodePtr> queue;
  queue.push(output);
  while (!queue.empty()) {
    NodePtr node = queue.front();
    queue.pop();
    if (visited.count(node) == 0) {
      visited.insert(node);
      for (Node input : node->input_nodes) {
        queue.push(input.get());
      }
    }
  }
  return visited;
}

// static
std::unordered_set<NodePtr> NodeImpl::RelevantNodes(NodePtr input,
                                                    NodePtr output) {
  std::unordered_set<NodePtr> forward_nodes = ForwardNodes(input);
  std::unordered_set<NodePtr> backward_nodes = BackwardNodes(output);
  std::unordered_set<NodePtr> relevant_nodes;
  for (NodePtr node : forward_nodes) {
    if (backward_nodes.count(node) > 0) {
      relevant_nodes.insert(node);
    }
  }
  return relevant_nodes;
}
