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

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/random.hpp"

using namespace std;
using namespace ngraph;

vector<shared_ptr<runtime::Tensor>> create_args(runtime::Backend& backend, const Function& f)
{
    vector<shared_ptr<runtime::Tensor>> args;
    vector<bool> args_cacheable;
    for (shared_ptr<op::Parameter> param : f.get_parameters())
    {
        auto tensor = backend.create_tensor(param->get_element_type(), param->get_shape());
        test::random_init(tensor.get());
        args.push_back(tensor);
    }
    return args;
}

vector<shared_ptr<runtime::Tensor>> create_results(runtime::Backend& backend, const Function& f)
{
    vector<shared_ptr<runtime::Tensor>> results;
    for (shared_ptr<Node> out : f.get_results())
    {
        auto result = backend.create_tensor(out->get_element_type(), out->get_shape());
        results.push_back(result);
    }
    return results;
}

TEST(backend_api, registered_devices)
{
    vector<string> devices = runtime::Backend::get_registered_devices();
    EXPECT_GE(devices.size(), 0);

    EXPECT_TRUE(find(devices.begin(), devices.end(), "INTERPRETER") != devices.end());
}

TEST(backend_api, invalid_name)
{
    ASSERT_ANY_THROW(ngraph::runtime::Backend::create("COMPLETELY-BOGUS-NAME"));
}

TEST(backend_api, performance)
{
    stopwatch timer;

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("NOP");

    {
        string json = file_util::path_join(CURDIR, "models", "mxnet", "LSTM_forward.json");
        auto f = deserialize(json);
        auto args_src = create_args(*backend, *f);
        auto results_src = create_results(*backend, *f);
        backend->compile(f);
        timer.start();
        vector<shared_ptr<runtime::Tensor>> args;
        for (const shared_ptr<runtime::Tensor>& t : args_src)
        {
            args.push_back(t);
        }
        vector<shared_ptr<runtime::Tensor>> results;
        for (const shared_ptr<runtime::Tensor>& t : results_src)
        {
            results.push_back(t);
        }

        backend->call_with_validate(f, results, args);
        timer.stop();
        cout << "fprop call time " << timer.get_microseconds() << "us" << endl;
    }

    {
        string json = file_util::path_join(CURDIR, "models", "mxnet", "LSTM_backward.json");
        auto f = deserialize(json);
        auto args = create_args(*backend, *f);
        auto results = create_results(*backend, *f);
        NGRAPH_INFO << args.size() << ", " << results.size();
        backend->compile(f);
        timer.start();
        vector<runtime::Tensor*> t_args;
        for (const shared_ptr<runtime::Tensor>& t : args)
        {
            t_args.push_back(t.get());
        }
        vector<runtime::Tensor*> t_results;
        for (const shared_ptr<runtime::Tensor>& t : results)
        {
            t_results.push_back(t.get());
        }
        backend->call_with_validate(f, results, args);
        timer.stop();
        cout << "bprop call time " << timer.get_microseconds() << "us" << endl;
    }
}
