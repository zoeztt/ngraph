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

#include <algorithm>

#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/ccpu/ccpu_call_frame.hpp"
#include "ngraph/runtime/ccpu/ccpu_external_function.hpp"
#include "ngraph/runtime/ccpu/ccpu_tensor_view.hpp"

using namespace std;
using namespace ngraph;

runtime::ccpu::CCPUCallFrame::CCPUCallFrame(std::shared_ptr<CCPUExternalFunction> external_function,
                                            EntryPoint compiled_function)
    : m_external_function(external_function)
    , m_compiled_function(compiled_function)
{
    setup_runtime_context();
}

runtime::ccpu::CCPUCallFrame::~CCPUCallFrame()
{
    cleanup_runtime_context();
}

void runtime::ccpu::CCPUCallFrame::call(
    const std::vector<std::shared_ptr<runtime::TensorView>>& output_tvs,
    const std::vector<std::shared_ptr<runtime::TensorView>>& input_tvs)
{
    vector<void*> inputs;
    vector<void*> outputs;

    propagate_layouts(output_tvs, m_external_function->get_result_layout_descriptors());

    for (size_t i = 0; i < input_tvs.size(); i++)
    {
        shared_ptr<runtime::ccpu::CCPUTensorView> tv =
            static_pointer_cast<runtime::ccpu::CCPUTensorView>(input_tvs[i]);
        ctx->p_en[i] = tv->get_stale();
        inputs.push_back(tv->get_data_ptr());
    }
    for (size_t i = 0; i < output_tvs.size(); i++)
    {
        shared_ptr<runtime::ccpu::CCPUTensorView> tv =
            static_pointer_cast<runtime::ccpu::CCPUTensorView>(output_tvs[i]);
        outputs.push_back(tv->get_data_ptr());
    }

    // Invoke compiled computation
    m_compiled_function(inputs.data(), outputs.data(), ctx);
}

void runtime::ccpu::CCPUCallFrame::propagate_layouts(
    const std::vector<std::shared_ptr<runtime::TensorView>>& tvs,
    const LayoutDescriptorPtrs& layouts) const
{
    if (layouts.size() != tvs.size())
    {
        throw ngraph_error(
            "Error propagating layouts - tensor view and layout descriptor counts do not match");
    }
    for (size_t i = 0; i < tvs.size(); i++)
    {
        if (layouts[i] == nullptr)
        {
            throw ngraph_error(
                "Error propagating layouts - layout information missing from tensor view");
        }
        tvs[i]->get_descriptor()->set_tensor_view_layout(layouts[i]);
    }
}

void runtime::ccpu::CCPUCallFrame::setup_runtime_context()
{
    ctx = new CCPURuntimeContext;

    ctx->p_en = new bool[m_external_function->get_parameter_layout_descriptors().size()];

    // Create temporary buffer pools
    size_t alignment = runtime::ccpu::CCPUExternalFunction::s_memory_pool_alignment;
    for (auto buffer_size : m_external_function->get_memory_buffer_sizes())
    {
        auto buffer = new AlignedBuffer(buffer_size, alignment);
        ctx->memory_buffers.push_back(buffer);
    }
    const auto& mkldnn_emitter = m_external_function->get_mkldnn_emitter();
    ctx->mkldnn_primitives = mkldnn_emitter->get_mkldnn_primitives().data();
    ctx->mkldnn_workspaces = mkldnn_emitter->get_mkldnn_workspaces().data();
}

void runtime::ccpu::CCPUCallFrame::cleanup_runtime_context()
{
    delete[] ctx->p_en;
    for (auto buffer : ctx->memory_buffers)
    {
        delete buffer;
    }
    delete ctx;
}
