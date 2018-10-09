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

#include "ngraph/pass/quantization.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(cpu_fusion, remove_dq)
{
    //const string json_path = file_util::path_join(SERIALIZED_ZOO, "tf_conv_mnist_nhwc.json");
    //const string json_string = file_util::read_file_to_string(json_path);
    const string json_string =
        file_util::read_file_to_string("tf_function_ngraph_cluster_0__4_trained.json");
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::Quantization>();
    //pass_manager.register_pass<pass::CoreFusion>();
    //pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    //pass_manager.register_pass<pass::VisualizeTree>("after.pdf");
    pass_manager.run_passes(func);

    //size_t before_after = count_ops_of_type<op::Reshape>(func);
    //ASSERT_LE(before_after, before_count);
}
