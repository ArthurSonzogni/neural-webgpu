#include "Node.hpp"
#include "Tensor.hpp"

Node Input(GPU& gpu, std::vector<int> sizes) {
  class Impl : public NodeImpl {
   public:
    Impl(GPU& gpu, std::vector<int> sizes) : NodeImpl(gpu) {
      outputs = {
        Tensor(sizes)
      };
      outputs_gradients = {
        Tensor(sizes)
      };
      outputs[0].Fill(gpu, 0.f);
      outputs_gradients[0].Fill(gpu, 0.f);
    }

    void Forward() override {
      // Do nothing.
    }

    void Backward() override {
      // Do nothing.
    }
  };
  return std::make_shared<Impl>(gpu, sizes);
}
