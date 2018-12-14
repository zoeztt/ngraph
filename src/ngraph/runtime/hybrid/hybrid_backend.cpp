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

#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/hybrid/hybrid_util.hpp"
#include "ngraph/runtime/hybrid/pass/assign_placement.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/runtime/gpu/gpu_tensor.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"

using namespace ngraph;
using namespace std;

namespace
{
    string get_placement(const runtime::Tensor* t)
    {
        string rc;
        if (dynamic_cast<const runtime::HostTensor*>(t) != nullptr)
        {
            rc = "HostTensor";
        }
        else if (dynamic_cast<const runtime::gpu::GPUTensor*>(t) != nullptr)
        {
            rc = "GPUTensor";
        }
        return rc;
    }
    string get_placement(const runtime::Backend* t)
    {
        string rc;
        if (dynamic_cast<const runtime::interpreter::INTBackend*>(t) != nullptr)
        {
            rc = "INTBackend";
        }
        else if (dynamic_cast<const runtime::gpu::GPU_Backend*>(t) != nullptr)
        {
            rc = "GPU_Backend Backend";
        }
        return rc;
    }
}

runtime::hybrid::HybridBackend::HybridBackend(
    const std::vector<std::shared_ptr<runtime::Backend>>& backend_list)
    : m_backend_list{backend_list}
{
}

shared_ptr<runtime::Tensor>
    runtime::hybrid::HybridBackend::create_tensor(const element::Type& element_type,
                                                  const Shape& shape)
{
    auto it = m_backend_list.begin();
    return (*it)->create_tensor(element_type, shape);
}

shared_ptr<runtime::Tensor> runtime::hybrid::HybridBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    auto it = m_backend_list.begin();
    return (*it)->create_tensor(element_type, shape, memory_pointer);
}

runtime::Handle runtime::hybrid::HybridBackend::compile(shared_ptr<Function> func)
{
    if (m_function_map.find(func) == m_function_map.end())
    {
        // Clone function
        FunctionInstance instance;
        instance.m_function = clone_function(*func);

        // Run placement pass
        ngraph::pass::Manager pass_manager;
        pass_manager.register_pass<runtime::hybrid::pass::AssignPlacement>(m_backend_list);
        pass_manager.run_passes(instance.m_function);

        // Split function to sub_functions
        tie(instance.m_sub_functions, instance.m_map_parameter_to_result) =
            split_function_by_placement_size(instance.m_function);
        m_function_map.insert({func, instance});

        // Compile subfunctions in corresponding backends
        for (shared_ptr<Function>& sub_function : instance.m_sub_functions)
        {
            size_t placement = get_colocated_function_placement_size(sub_function);
            auto backend = m_backend_list[placement];
            backend->compile(sub_function);

            // Compile will replace nodes so we need to make one more pass through all
            // ops to reset placement
            for (auto op : sub_function->get_ops())
            {
                op->set_placement_index(placement);
            }
        }
    }

    return func;
}

bool runtime::hybrid::HybridBackend::call(shared_ptr<Function> func,
                                          const vector<shared_ptr<runtime::Tensor>>& outputs,
                                          const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    // Get FunctionInstance
    bool rc = true;

    NGRAPH_INFO;
    auto it = m_function_map.find(func);
    if (it == m_function_map.end())
    {
        throw runtime_error("compile() must be called before call().");
    }
    FunctionInstance& instance = it->second;

    NGRAPH_INFO;
    // Parameter and result node in sub_function maps to one Tensor
    unordered_map<shared_ptr<Node>, shared_ptr<runtime::Tensor>> map_node_to_tensor;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        NGRAPH_INFO << get_placement(inputs[i].get());
        map_node_to_tensor[instance.m_function->get_parameters()[i]] = inputs[i];
    }
    NGRAPH_INFO;
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        NGRAPH_INFO << get_placement(outputs[i].get());
        map_node_to_tensor[instance.m_function->get_results()[i]] = outputs[i];
    }

    NGRAPH_INFO;
    // Call subfunctions
    for (const shared_ptr<Function>& sub_function : instance.m_sub_functions)
    {
        // Init backend
        size_t placement = get_colocated_function_placement_size(sub_function);
        auto backend = m_backend_list[placement];
        NGRAPH_INFO << get_placement(backend.get());

        // Prepare parameter Tensors
        NGRAPH_INFO;
        vector<shared_ptr<runtime::Tensor>> parameters;
        for (auto parameter_node : sub_function->get_parameters())
        {
            auto it = map_node_to_tensor.find(parameter_node);
            if (it != map_node_to_tensor.end())
            {
                parameters.push_back(it->second);
            }
            else
            {
                auto result_node = instance.m_map_parameter_to_result.at(parameter_node);
                auto result = map_node_to_tensor.at(result_node);
                auto parameter = backend->create_tensor(parameter_node->get_element_type(),
                                                        parameter_node->get_shape());
                parameter->copy_from(*result);
                map_node_to_tensor[parameter_node] = parameter;
                parameters.push_back(parameter);
            }
        }

        // Prepare result Tensors
        NGRAPH_INFO;
        vector<shared_ptr<runtime::Tensor>> results;
        for (auto result_node : sub_function->get_results())
        {
            auto it = map_node_to_tensor.find(result_node);
            if (it != map_node_to_tensor.end())
            {
                results.push_back(it->second);
            }
            else
            {
                auto result = backend->create_tensor(result_node->get_element_type(),
                                                     result_node->get_shape());
                map_node_to_tensor[result_node] = result;
                results.push_back(result);
            }
        }

        // Call
        NGRAPH_INFO;
        backend->call(sub_function, results, parameters);
        NGRAPH_INFO;
    }
    NGRAPH_INFO;
    return rc;
}

bool runtime::hybrid::HybridBackend::is_supported(const Node& node) const
{
    return true;
}
