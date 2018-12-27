//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <stdexcept>

#include "ngraph/ngraph.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// Abs,
// Acos,
// ...
#define NGRAPH_OP(a, b) a,
enum class OP_TYPEID
{
#include "ngraph/op/op_tbl.hpp"
};
#undef NGRAPH_OP

/// \brief This class allows adding an enum typeid to each Node. This makes dealing with
/// collections of Nodes a little easier and faster as we can use switch() instead of
/// if/else statements
class NodeWrapper
{
public:
    NodeWrapper(const std::shared_ptr<const ngraph::Node>& node);

    const ngraph::Node& get_node() const { return *m_node; }
    OP_TYPEID get_typeid() const { return m_typeid; }
private:
    std::shared_ptr<const ngraph::Node> m_node;
    OP_TYPEID m_typeid;
};

class TestBackend : public ngraph::runtime::hybrid::HybridBackend
{
public:
    TestBackend();
};

class TestBackendImplementation : public ngraph::runtime::Backend
{
public:
    std::shared_ptr<ngraph::runtime::Tensor> create_tensor(const ngraph::element::Type& type,
                                                           const ngraph::Shape& shape,
                                                           void* memory_pointer) override;

    std::shared_ptr<ngraph::runtime::Tensor> create_tensor(const ngraph::element::Type& type,
                                                           const ngraph::Shape& shape) override;

    ngraph::runtime::Handle compile(std::shared_ptr<ngraph::Function> function) override;

    bool call(std::shared_ptr<ngraph::Function> function,
              const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& outputs,
              const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& intputs) override;

    bool is_supported(const ngraph::Node& node) const override;

private:
    int get_alignment() const { return 64; }
    class FunctionInstance
    {
    public:
        bool m_is_compiled = false;
        std::vector<NodeWrapper> m_wrapped_nodes;
        // std::unordered_map<const ngraph::Node*, std::shared_ptr<RNGState>> m_states;
        std::shared_ptr<ngraph::runtime::AlignedBuffer> m_temporary_memory;

        void* get_temporary_pointer(size_t offset) { return m_temporary_memory->get_ptr(offset); }
    };
    std::map<std::shared_ptr<ngraph::Function>, FunctionInstance> m_function_map;

    void generate_calls(const ngraph::element::Type& type,
                        const NodeWrapper& op,
                        const std::vector<void*>& outputs,
                        const std::vector<const void*>& inputs,
                        FunctionInstance& instance);

    template <typename T>
    void op_engine(const NodeWrapper& node_wrapper,
                   const std::vector<void*>& out,
                   const std::vector<const void*>& args,
                   FunctionInstance& instance)
    {
        const ngraph::Node& node = node_wrapper.get_node();

        switch (node_wrapper.get_typeid())
        {
        case OP_TYPEID::Add: { break;
        }
        default: NGRAPH_INFO << "Unsupported op '" << node.description() << "'";
        }
    }
};
