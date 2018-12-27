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

#include "gtest/gtest.h"

#include "hybrid_utils.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

// static runtime::Backend* hybrid1_creator(const char* config)
// {
//     vector<shared_ptr<runtime::Backend>> backend_list;
//     set<string> s0 = {"Add"};
//     auto b0 = make_shared<BackendWrapper>("INTERPRETER", s0, "AddOnly");
//     backend_list.push_back(b0);

// #define NGRAPH_OP(a, b) #a,
//     set<string> s1 = {
// #include "ngraph/op/op_tbl.hpp"
//     };
//     auto b1 = make_shared<BackendWrapper>("INTERPRETER", s1, "AllOps");
//     backend_list.push_back(b1);

//     return new TestBackend(backend_list);
// }

// TEST(HYBRID, abc)
// {
//     const string backend_name = "HYBRID1";
//     runtime::BackendManager::register_backend(backend_name, hybrid1_creator);

//     Shape shape{2, 2};
//     auto A = make_shared<op::Parameter>(element::f32, shape);
//     auto B = make_shared<op::Parameter>(element::f32, shape);
//     auto C = make_shared<op::Parameter>(element::f32, shape);
//     auto f = make_shared<Function>((A + B) * C, ParameterVector{A, B, C});

//     auto backend = runtime::Backend::create(backend_name);

//     // Create some tensors for input/output
//     shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
//     shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
//     shared_ptr<runtime::Tensor> c = backend->create_tensor(element::f32, shape);
//     shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

//     copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
//     copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
//     copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

//     auto handle = backend->compile(f);
//     backend->call_with_validate(handle, {result}, {a, b, c});
//     EXPECT_EQ(read_vector<float>(result),
//               (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

//     backend->call_with_validate(handle, {result}, {b, a, c});
//     EXPECT_EQ(read_vector<float>(result),
//               (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

//     backend->call_with_validate(handle, {result}, {a, c, b});
//     EXPECT_EQ(read_vector<float>(result),
//               (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());
// }

TEST(HYBRID, temp_passthrough)
{
    TestBackend::Register();

    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = make_shared<op::Parameter>(element::f32, shape);
    auto t1 = A + B;
    auto t2 = C * D;
    auto t3 = t1 + t2;
    auto f = make_shared<Function>(t2 + t3, ParameterVector{A, B, C, D});

    auto backend = runtime::Backend::create("HYBRID1");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> d = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, vector<float>({1, 2, 3, 4, 5, 6}));
    copy_data(b, vector<float>({2, 3, 4, 5, 6, 7}));
    copy_data(c, vector<float>({3, 4, 5, 6, 7, 8}));
    copy_data(d, vector<float>({4, 5, 6, 7, 8, 9}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c, d});
    EXPECT_EQ(read_vector<float>(result), vector<float>({27, 45, 67, 93, 123, 157}));
}
