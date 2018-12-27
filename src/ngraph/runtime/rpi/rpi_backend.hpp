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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/runtime/rpi/kernel/broadcast.hpp"
#include "ngraph/runtime/rpi/kernel/dot.hpp"
#include "ngraph/runtime/rpi/kernel/reshape.hpp"
#include "ngraph/runtime/rpi/kernel/result.hpp"
#include "ngraph/runtime/rpi/node_wrapper.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/state/rng_state.hpp"
#include "ngraph/util.hpp"

#ifdef NGRAPH_DISTRIBUTED
#include "ngraph/runtime/reference/allreduce.hpp"
#endif

namespace ngraph
{
    namespace runtime
    {
        namespace rpi
        {
            class RPIBackend;
            class RPIBackendOverride;
        }
    }
}

class ngraph::runtime::rpi::RPIBackend : public ngraph::runtime::hybrid::HybridBackend
{
public:
    RPIBackend();
};

class ngraph::runtime::rpi::RPIBackendOverride : public Backend
{
public:
    std::shared_ptr<Tensor>
        create_tensor(const element::Type& type, const Shape& shape, void* memory_pointer) override;

    std::shared_ptr<Tensor> create_tensor(const element::Type& type, const Shape& shape) override;

    Handle compile(std::shared_ptr<Function> function) override;

    bool call(std::shared_ptr<Function> function,
              const std::vector<std::shared_ptr<Tensor>>& outputs,
              const std::vector<std::shared_ptr<Tensor>>& intputs) override;

    void enable_performance_data(std::shared_ptr<Function> func, bool enable) override;
    std::vector<PerformanceCounter>
        get_performance_data(std::shared_ptr<Function> func) const override;

    bool is_supported(const Node& node) const override;

private:
    int get_alignment() const { return 64; }
    class FunctionInstance
    {
    public:
        bool m_is_compiled = false;
        bool m_nan_check_enabled = false;
        bool m_performance_counters_enabled = false;
        std::unordered_map<const Node*, stopwatch> m_timer_map;
        std::vector<NodeWrapper> m_wrapped_nodes;
        std::unordered_map<const Node*, std::shared_ptr<RNGState>> m_states;
        std::shared_ptr<AlignedBuffer> m_temporary_memory;

        void* get_temporary_pointer(size_t offset) { return m_temporary_memory->get_ptr(offset); }
    };
    std::map<std::shared_ptr<Function>, FunctionInstance> m_function_map;

    void generate_calls(const element::Type& type,
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
        const Node& node = node_wrapper.get_node();

        switch (node_wrapper.get_typeid())
        {
        case OP_TYPEID::Broadcast:
        {
            const op::Broadcast* broadcast = static_cast<const op::Broadcast*>(&node);
            const Shape& in_shape = node.get_input_shape(0);
            const Shape& out_shape = node.get_output_shape(0);
            const AxisSet& broadcast_axes = broadcast->get_broadcast_axes();
            rpi::kernel::broadcast(static_cast<const T*>(args[0]),
                                   static_cast<T*>(out[0]),
                                   in_shape,
                                   out_shape,
                                   broadcast_axes);
            break;
        }
        case OP_TYPEID::Dot:
        {
            const op::Dot* dot = static_cast<const op::Dot*>(&node);
            rpi::kernel::dot(static_cast<const T*>(args[0]),
                             static_cast<const T*>(args[1]),
                             static_cast<T*>(out[0]),
                             node.get_input_shape(0),
                             node.get_input_shape(1),
                             node.get_output_shape(0),
                             dot->get_reduction_axes_count());
            break;
        }
        case OP_TYPEID::Reshape:
        {
            const op::Reshape* reshape = static_cast<const op::Reshape*>(&node);
            // reference::reshape(static_cast<const T*>(args[0]),
            //                    static_cast<T*>(out[0]),
            //                    node.get_input_shape(0),
            //                    reshape->get_input_order(),
            //                    node.get_output_shape(0));
            rpi::kernel::reshape<T>(static_cast<const T*>(args[0]),
                                    static_cast<T*>(out[0]),
                                    node.get_input_shape(0),
                                    reshape->get_input_order(),
                                    node.get_output_shape(0));
            break;
        }
        case OP_TYPEID::Result:
        {
            const op::Result* res = static_cast<const op::Result*>(&node);
            rpi::kernel::result(static_cast<const T*>(args[0]),
                                static_cast<T*>(out[0]),
                                shape_size(res->get_shape()));
            break;
        }
        default: throw unsupported_op("Unsupported op '" + node.description() + "'");
        }
    }
};
