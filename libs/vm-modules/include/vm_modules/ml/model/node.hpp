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
  using GraphType    = fetch::ml::Graph<TensorType>;
  using VMTensor     = fetch::vm_modules::math::VMTensor;
  using NodePtrType  = typename std::shared_ptr<fetch::ml::Node<TensorType>>;

  VMNode(VMNode const &other) = delete;
  VMNode(VMNode &&other)      = delete;
  VMNode &operator=(VMNode const &other) = default;
  VMNode &operator=(VMNode &&other) = delete;

  VMNode(fetch::vm::VM *vm, fetch::vm::TypeId type_id);

  VMNode(fetch::vm::VM *vm, fetch::vm::TypeId type_id,
         fetch::vm::Ptr<fetch::vm::String> const &layer_category);

  VMNode(fetch::vm::VM *vm, fetch::vm::TypeId type_id, std::string const &layer_category);

  static fetch::vm::Ptr<VMNode> Constructor(
      fetch::vm::VM *vm, fetch::vm::TypeId type_id,
      fetch::vm::Ptr<fetch::vm::String> const &layer_category);

  static void Bind(fetch::vm::Module &module);

private:
  NodePtrType        node_;
  SupportedLayerType layer_category_ = SupportedLayerType::INPUT;

  void Init(std::string const &layer_category);
};

using namespace fetch::vm;

VMNode::VMNode(VM *vm, TypeId type_id)
  : Object(vm, type_id)
  , estimator_{*this}
{
  Init("none");
}

VMNode::VMNode(VM *vm, TypeId type_id, fetch::vm::Ptr<fetch::vm::String> const &layer_category)
  : Object(vm, type_id)
  , estimator_{*this}
{
  Init(layer_category->string());
}

VMNode::VMNode(VM *vm, TypeId type_id, std::string const &layer_category)
  : Object(vm, type_id)
  , estimator_{*this}
{
  Init(layer_category);
}

void VMNode::Init(std::string const &layer_category)
{
  node_ = NodePtrType(fetch::ml::OpType::OP_PLACEHOLDER, layer_category,
                      []() { return std::make_shared<fetch::ml::ops::PlaceHolder<TypeParam>>(); });
}

Ptr<VMNode> VMNode::Constructor(VM *vm, TypeId type_id,
                                fetch::vm::Ptr<fetch::vm::String> const &layer_category)
{
  return Ptr<VMNode>{new VMNode(vm, type_id, layer_category)};
}

void VMNode::Bind(Module &module)
{
  static const ChargeAmount FIXED_CONSTRUCTION_CHARGE{100};
  module.CreateClassType<VMNode>("Node")
      .CreateConstructor(&VMNode::Constructor, FIXED_CONSTRUCTION_CHARGE)
      .CreateSerializeDefaultConstructor([](VM *vm, TypeId type_id) -> Ptr<VMNode> {
        return Ptr<VMNode>{new VMNode(vm, type_id)};
      });
}

}  // namespace model
}  // namespace ml
}  // namespace vm_modules
}  // namespace fetch
