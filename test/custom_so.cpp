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

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "ngraph/runtime/tensor_view.hpp"

using namespace std;
using namespace ngraph;

extern "C" void execute(void* user_data,
                        runtime::Backend* backend,
                        const std::vector<std::shared_ptr<runtime::TensorView>>& out,
                        const std::vector<std::shared_ptr<runtime::TensorView>>& args)
{
    if (dynamic_cast<runtime::interpreter::INTBackend*>(backend))
    {
        const float* arg0 =
            dynamic_pointer_cast<runtime::HostTensorView>(args[0])->get_data_ptr<float>();
        const float* arg1 =
            dynamic_pointer_cast<runtime::HostTensorView>(args[1])->get_data_ptr<float>();
        const float* arg2 =
            dynamic_pointer_cast<runtime::HostTensorView>(args[2])->get_data_ptr<float>();
        float* out0 = dynamic_pointer_cast<runtime::HostTensorView>(out[0])->get_data_ptr<float>();
        size_t size = out[0]->get_element_count();
        for (size_t i = 0; i < size; i++)
        {
            out0[i] = (arg0[i] + arg1[i]) * arg2[i];
        }
    }
}
