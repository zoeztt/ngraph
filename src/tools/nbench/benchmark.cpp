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

#include <random>

#include "benchmark.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static default_random_engine s_random_engine;

template <>
void get_range(char& lower, char& upper)
{
    lower = 0;
    upper = 1;
}

template <>
void get_range(float& lower, float& upper)
{
    lower = -1;
    upper = 1;
}

template <>
void get_range(double& lower, double& upper)
{
    lower = -1;
    upper = 1;
}

template <>
void get_range(int8_t& lower, int8_t& upper)
{
    lower = -1;
    upper = 1;
}

template <>
void get_range(int16_t& lower, int16_t& upper)
{
    lower = -1;
    upper = 1;
}

template <>
void get_range(int32_t& lower, int32_t& upper)
{
    lower = 0;
    upper = 1;
}

template <>
void get_range(int64_t& lower, int64_t& upper)
{
    lower = -1;
    upper = 1;
}

template <>
void get_range(uint8_t& lower, uint8_t& upper)
{
    lower = 0;
    upper = 1;
}

template <>
void get_range(uint16_t& lower, uint16_t& upper)
{
    lower = 0;
    upper = 1;
}

template <>
void get_range(uint32_t& lower, uint32_t& upper)
{
    lower = 0;
    upper = 1;
}

template <>
void get_range(uint64_t& lower, uint64_t& upper)
{
    lower = 0;
    upper = 1;
}

template <typename T>
void init_int_tv(shared_ptr<runtime::Tensor> tv)
{
    T lower;
    T upper;
    get_range<T>(lower, upper);
    size_t size = tv->get_element_count();
    uniform_int_distribution<T> dist(lower, upper);
    vector<T> vec(size);
    for (T& element : vec)
    {
        element = dist(s_random_engine);
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(T));
}

template <>
void init_int_tv<char>(shared_ptr<runtime::Tensor> tv)
{
    char lower;
    char upper;
    get_range<char>(lower, upper);
    size_t size = tv->get_element_count();
    uniform_int_distribution<int16_t> dist(static_cast<short>(lower), static_cast<short>(upper));
    vector<char> vec(size);
    for (char& element : vec)
    {
        element = static_cast<char>(dist(s_random_engine));
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(char));
}

template <>
void init_int_tv<int8_t>(shared_ptr<runtime::Tensor> tv)
{
    int8_t lower;
    int8_t upper;
    get_range<int8_t>(lower, upper);
    size_t size = tv->get_element_count();
    uniform_int_distribution<int16_t> dist(static_cast<short>(lower), static_cast<short>(upper));
    vector<int8_t> vec(size);
    for (int8_t& element : vec)
    {
        element = static_cast<int8_t>(dist(s_random_engine));
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(int8_t));
}

template <>
void init_int_tv<uint8_t>(shared_ptr<runtime::Tensor> tv)
{
    uint8_t lower;
    uint8_t upper;
    get_range<uint8_t>(lower, upper);
    size_t size = tv->get_element_count();
    uniform_int_distribution<int16_t> dist(static_cast<short>(lower), static_cast<short>(upper));
    vector<uint8_t> vec(size);
    for (uint8_t& element : vec)
    {
        element = static_cast<uint8_t>(dist(s_random_engine));
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(uint8_t));
}

template <typename T>
void init_real_tv(shared_ptr<runtime::Tensor> tv)
{
    T lower;
    T upper;
    get_range<T>(lower, upper);
    size_t size = tv->get_element_count();
    uniform_real_distribution<T> dist(lower, upper);
    vector<T> vec(size);
    for (T& element : vec)
    {
        element = dist(s_random_engine);
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(T));
}

static void random_init(shared_ptr<runtime::Tensor> tv)
{
    element::Type et = tv->get_element_type();
    if (et == element::boolean)
    {
        init_int_tv<char>(tv);
    }
    else if (et == element::f32)
    {
        init_real_tv<float>(tv);
    }
    else if (et == element::f64)
    {
        init_real_tv<double>(tv);
    }
    else if (et == element::i8)
    {
        init_int_tv<int8_t>(tv);
    }
    else if (et == element::i16)
    {
        init_int_tv<int16_t>(tv);
    }
    else if (et == element::i32)
    {
        init_int_tv<int32_t>(tv);
    }
    else if (et == element::i64)
    {
        init_int_tv<int64_t>(tv);
    }
    else if (et == element::u8)
    {
        init_int_tv<uint8_t>(tv);
    }
    else if (et == element::u16)
    {
        init_int_tv<uint16_t>(tv);
    }
    else if (et == element::u32)
    {
        init_int_tv<uint32_t>(tv);
    }
    else if (et == element::u64)
    {
        init_int_tv<uint64_t>(tv);
    }
    else
    {
        throw runtime_error("unsupported type");
    }
}

vector<runtime::PerformanceCounter> run_benchmark(shared_ptr<Function> f,
                                                  const string& backend_name,
                                                  size_t iterations,
                                                  bool timing_detail,
                                                  int warmup_iterations,
                                                  bool copy_data)
{
    stopwatch timer;
    timer.start();
    auto backend = runtime::Backend::create(backend_name);
    backend->enable_performance_data(f, timing_detail);
    backend->compile(f);
    timer.stop();
    cout.imbue(locale(""));
    cout << "compile time: " << timer.get_milliseconds() << "ms" << endl;

    vector<shared_ptr<runtime::HostTensor>> arg_data;
    vector<shared_ptr<runtime::Tensor>> args;
    vector<bool> args_cacheable;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        auto tensor = backend->create_tensor(param->get_element_type(), param->get_shape());
        auto tensor_data =
            make_shared<runtime::HostTensor>(param->get_element_type(), param->get_shape());
        random_init(tensor);
        args.push_back(tensor);
        arg_data.push_back(tensor_data);
        args_cacheable.push_back(param->get_cacheable());
    }
    vector<shared_ptr<runtime::HostTensor>> result_data;
    vector<shared_ptr<runtime::Tensor>> results;
    for (shared_ptr<Node> out : f->get_results())
    {
        auto result = backend->create_tensor(out->get_element_type(), out->get_shape());
        auto tensor_data =
            make_shared<runtime::HostTensor>(out->get_element_type(), out->get_shape());
        results.push_back(result);
        result_data.push_back(tensor_data);
    }

    for (size_t i = 0; i < args.size(); i++)
    {
        if (args_cacheable[i])
        {
            args[i]->set_stale(false);
        }
    }

    if (warmup_iterations)
    {
        for (int i = 0; i < warmup_iterations; i++)
        {
            backend->call(f, results, args);
        }
    }

    stopwatch t1;
    t1.start();
    for (size_t i = 0; i < iterations; i++)
    {
        if (copy_data)
        {
            for (size_t arg_index = 0; arg_index < args.size(); arg_index++)
            {
                const shared_ptr<runtime::Tensor>& arg = args[arg_index];
                if (arg->get_stale())
                {
                    const shared_ptr<runtime::HostTensor>& data = arg_data[arg_index];
                    arg->write(data->get_data_ptr(),
                               0,
                               data->get_element_count() * data->get_element_type().size());
                }
            }
        }
        backend->call(f, results, args);
        if (copy_data)
        {
            for (size_t result_index = 0; result_index < results.size(); result_index++)
            {
                const shared_ptr<runtime::HostTensor>& data = result_data[result_index];
                const shared_ptr<runtime::Tensor>& result = results[result_index];
                result->read(data->get_data_ptr(),
                             0,
                             data->get_element_count() * data->get_element_type().size());
            }
        }
    }
    t1.stop();
    float time = t1.get_milliseconds();
    cout << time / iterations << "ms per iteration" << endl;

    vector<runtime::PerformanceCounter> perf_data = backend->get_performance_data(f);
    return perf_data;
}

template <typename T>
void get_range_string(string& lower, string& upper)
{
    T l;
    T u;
    get_range<T>(l, u);
    lower = to_string(l);
    upper = to_string(u);
}

void get_range_string(const Node* n, string& lower, string& upper)
{
    element::Type et = n->get_element_type();
    if (et == element::boolean)
    {
        get_range_string<char>(lower, upper);
    }
    else if (et == element::f32)
    {
        get_range_string<float>(lower, upper);
    }
    else if (et == element::f64)
    {
        get_range_string<double>(lower, upper);
    }
    else if (et == element::i8)
    {
        get_range_string<int8_t>(lower, upper);
    }
    else if (et == element::i16)
    {
        get_range_string<int16_t>(lower, upper);
    }
    else if (et == element::i32)
    {
        get_range_string<int32_t>(lower, upper);
    }
    else if (et == element::i64)
    {
        get_range_string<int64_t>(lower, upper);
    }
    else if (et == element::u8)
    {
        get_range_string<uint8_t>(lower, upper);
    }
    else if (et == element::u16)
    {
        get_range_string<uint16_t>(lower, upper);
    }
    else if (et == element::u32)
    {
        get_range_string<uint32_t>(lower, upper);
    }
    else if (et == element::u64)
    {
        get_range_string<uint64_t>(lower, upper);
    }
    else
    {
        throw runtime_error("unsupported type");
    }
}
