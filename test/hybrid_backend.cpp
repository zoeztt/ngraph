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

#include <memory>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/assign_placement.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/util.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

class TestBackend : public runtime::Backend
{
public:
    TestBackend() {}
    ~TestBackend() {}
    bool is_supported(const Node& node)
    {
        if (node.description() == "Add")
        {
            return true;
        }

        return false;
    }
    bool compile(const shared_ptr<Function>& func)
    {
        if (m_function_map.find(func) == m_function_map.end())
        {
            // Clone function
            FunctionInstance instance;
            instance.m_function = clone_function(*func);

            // Run placement pass
            pass::Manager pass_manager;
            pass_manager.run_passes(instance.m_function);
        }
        return true;
    }

    bool call_with_validate(const shared_ptr<Function>& func,
                            const vector<shared_ptr<runtime::Tensor>>& outputs,
                            const vector<shared_ptr<runtime::Tensor>>& inputs)
    {
        return true;
    }

protected:
    class FunctionInstance
    {
    public:
        shared_ptr<Function> m_function;
        vector<shared_ptr<Function>> m_sub_functions;
        unordered_map<shared_ptr<op::Parameter>, shared_ptr<op::Result>> m_map_parameter_to_result;
    };
    map<shared_ptr<Function>, FunctionInstance> m_function_map;
};
