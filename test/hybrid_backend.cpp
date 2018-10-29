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
#include "gtest/gtest.h"

#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/assign_placement.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/util.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

class TestBackend 
{
public:
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

TEST(Hybrid, abc)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    auto hybrid_backend = runtime::Backend::create("HYBRID");

    // // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = hybrid_backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = hybrid_backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> c = hybrid_backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = hybrid_backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    // auto test_backend = make_shared<TestBackend>();
    auto interpreted_1 = runtime::Backend::create("INTERPRETER");
    auto interpreted_2 = runtime::Backend::create("INTERPRETER");
    vector<shared_ptr<ngraph::runtime::Backend>>  backends ;
    backends.push_back(interpreted_1);
    backends.push_back(interpreted_2);

    shared_ptr<runtime::hybrid::HYBRIDBackend> hybackend =
        static_pointer_cast<runtime::hybrid::HYBRIDBackend>(hybrid_backend);

    auto status_compiled = hybackend->compile_for_backends(f,
                                          backends);
    // auto backend = runtime::Backend::create(TestBackend);
    // auto test_backend = new TestBackend;
    // shared_ptr<runtime::Backend> testbackend_intas = shared_ptr<runtime::Backend>(new TestBackend);
    //  auto test_backend = make_shared<TestBackend>();

    // vector<shared_ptr<runtime::Backend>> backend_vect; 
    
    // backend_vect.push_back(test_backend); 

    // backend->call_with_validate(f, {result}, {a, b, c});
    // EXPECT_EQ(read_vector<float>(result),
    //           (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    // backend->call_with_validate(f, {result}, {b, a, c});
    // EXPECT_EQ(read_vector<float>(result),
    //           (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    // backend->call_with_validate(f, {result}, {a, c, b});
    // EXPECT_EQ(read_vector<float>(result),
    //           (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());
}
