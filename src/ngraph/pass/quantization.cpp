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

#include <set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/util.hpp"
#include "quantization.hpp"

using namespace ngraph;
using namespace std;

bool ngraph::pass::Quantization::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    m_quantized_region_nodes.clear();

    for (auto n : f->get_ops())
    {
        //skip nodes that are already in quantized regions
        if (m_quantized_region_nodes.count(n) != 0)
        {
            continue;
        }
        if (auto quantize = std::dynamic_pointer_cast<op::Quantize>(n))
        {
            auto outputs = collect_quantized_region_outputs(n);
            NodeVector quantized_region;
            NodeVector dequantize_nodes;
            find_quantized_region(outputs, quantized_region, dequantize_nodes);
            m_quantized_region_nodes.insert(quantized_region.begin(), quantized_region.end());
            process_quantized_region(quantized_region, dequantize_nodes);
        }
    }
    return false;
}

void ngraph::pass::Quantization::find_quantized_region(const NodeVector& outputs,
                                                       NodeVector& quantized_region,
                                                       NodeVector& dequantize_nodes)
{
    std::unordered_set<std::shared_ptr<Node>> instances_seen;
    std::deque<std::shared_ptr<Node>> stack(outputs.begin(), outputs.end());
    while (stack.size() > 0)
    {
        auto n = stack.front();
        if (instances_seen.count(n) == 0)
        {
            // if (n->get_inputs().size() == 0)
            // {
            // 	return NodeVector{};
            // }
            instances_seen.insert(n);
            quantized_region.push_back(n);
        }
        stack.pop_front();

        if (std::dynamic_pointer_cast<op::Dequantize>(n))
        {
            dequantize_nodes.push_back(n);
            continue;
        }

        for (auto arg : n->get_arguments())
        {
            if (instances_seen.count(arg) == 0)
            {
                stack.push_front(arg);
            }
        }
    }
    return;
}

NodeVector ngraph::pass::Quantization::collect_quantized_region_outputs(std::shared_ptr<Node> n)
{
    if (auto goe = std::dynamic_pointer_cast<op::GetOutputElement>(n->get_argument(0)))
    {
        auto multi = goe->get_inputs().at(0).get_output().get_node();
        NodeVector outputs;
        for (auto u : multi->get_users())
        {
            for (auto q : u->get_users())
            {
                if (auto quantize = std::dynamic_pointer_cast<op::Quantize>(u))
                {
                    outputs.push_back(quantize);
                }
            }
        }
        return outputs;
    }
    else
    {
        return NodeVector{n};
    }
}

void ngraph::pass::Quantization::process_quantized_region(
    const ngraph::NodeVector& qr, const ngraph::NodeVector& dequantize_nodes)
{
    std::cout << "Quantized region:\n";
    for (auto n : qr)
    {
        NGRAPH_DEBUG << " n = " << n->get_name() << std::endl;
    }
}
