#pragma once
//------------------------------------------------------------------------------
//
//   Copyright 2018-2020 Fetch.AI Limited
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
//------------------------------------------------------------------------------

#include "ml/ops/placeholder.hpp"
#include "vm/module.hpp"
#include "vm/object.hpp"
#include "vm_modules/math/tensor/tensor.hpp"
#include "vm_modules/math/type.hpp"
#include "vm_modules/ml/model/model_estimator.hpp"

namespace fetch {

namespace vm {
class Module;
}

namespace vm_modules {
namespace ml {
namespace model {

enum class SupportedLayerType : uint8_t
{
  DENSE,
  INPUT
};

class VMNode : public fetch::vm::Object
{
  friend class fetch::vm_modules::ml::model::ModelEstimator;

public:
  using ChargeAmount = uint64_t;
  using TypeId       = uint16_t;
  using DataType     = fetch::vm_modules::math::DataType;
  using TensorType   = fetch::math::Tensor<DataType>;
  using VMTensor     = fetch::vm_modules::math::VMTensor;
  using NodePtrType  = typename std::shared_ptr<fetch::ml::Node<TensorType>>;
  using GraphType = typename fetch::ml::Graph<fetch::math::Tensor<DataType>>;
  using GraphPtrType = typename std::shared_ptr<GraphType>;

  VMNode(VMNode const &other) = delete;
  VMNode(VMNode &&other)      = delete;
  VMNode &operator=(VMNode const &other) = default;
  VMNode &operator=(VMNode &&other) = delete;

  NodePtrType &operator()(VMTensor &other)
  {
      return latest_node_; // trash, might not compile yet
  }

  NodePtrType &operator()(VMNode &node)
  {
      // node should be the pointer to the last node in the graph
      // add constructed node to the existing graph
      // return the pointer to the latest node in the graph
      auto node = NodePtrType(fetch::ml::OpType::OP_PLACEHOLDER, layer_type, []() { return std::make_shared<fetch::ml::ops::PlaceHolder<TypeParam>>(); });
      node->Add<fetch::ml::ops::Flatten<TensorType>>();
      return node;
  }

  VMNode(fetch::vm::VM *vm, fetch::vm::TypeId type_id);

  VMNode(fetch::vm::VM *vm, fetch::vm::TypeId type_id,
         fetch::vm::Ptr<fetch::vm::String> const &layer_type);

  VMNode(fetch::vm::VM *vm, fetch::vm::TypeId type_id, std::string const &layer_type);

  static fetch::vm::Ptr<VMNode> Constructor(
      fetch::vm::VM *vm, fetch::vm::TypeId type_id,
      fetch::vm::Ptr<fetch::vm::String> const &layer_type);

  static void Bind(fetch::vm::Module &module);

private:
  NodePtrType        latest_node_= nullptr;
  SupportedLayerType layer_type_ = SupportedLayerType::INPUT;
  GraphPtrType graph_ = nullptr;

  void Init(std::string const &layer_type);
};

using namespace fetch::vm;

VMNode::VMNode(VM *vm, TypeId type_id)
  : Object(vm, type_id)
  , estimator_{*this}
{
  Init("none");
}

VMNode::VMNode(VM *vm, TypeId type_id, fetch::vm::Ptr<fetch::vm::String> const &layer_type)
  : Object(vm, type_id)
  , estimator_{*this}
{
  Init(layer_type->string());
}

VMNode::VMNode(VM *vm, TypeId type_id, std::string const &layer_type)
  : Object(vm, type_id)
  , estimator_{*this}
{
  Init(layer_type);
}

void VMNode::Init(std::string const &layer_type)
{
    // here should be a switch case with layers i guess
    if(!graph_) {
        graph_ = std::make_shared<GraphType>();
    }
    graph_->template AddNode<fetch::ml::ops::PlaceHolder<TensorType>>(layer_type, {});
    latest_node_ = graph_->GetNode(layer_type);
}

Ptr<VMNode> VMNode::Constructor(VM *vm, TypeId type_id,
                                fetch::vm::Ptr<fetch::vm::String> const &layer_type)
{
  return Ptr<VMNode>{new VMNode(vm, type_id, layer_type)};
}

void VMNode::Bind(Module &module)
{
  static const ChargeAmount FIXED_CONSTRUCTION_CHARGE{33};
  module.CreateClassType<VMNode>("add")
      .CreateConstructor(&VMNode::Constructor, FIXED_CONSTRUCTION_CHARGE)
      .CreateSerializeDefaultConstructor([](VM *vm, TypeId type_id) -> Ptr<VMNode> {
        return Ptr<VMNode>{new VMNode(vm, type_id)};
      });
}

}  // namespace model
}  // namespace ml
}  // namespace vm_modules
}  // namespace fetch
