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

#include "default_values.hpp"

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
void init_int_tv(runtime::Tensor& tv)
{
    T lower;
    T upper;
    get_range<T>(lower, upper);
    size_t size = tv.get_element_count();
    uniform_int_distribution<T> dist(lower, upper);
    vector<T> vec(size);
    for (T& element : vec)
    {
        element = dist(s_random_engine);
    }
    tv.write(vec.data(), 0, vec.size() * sizeof(T));
}

template <>
void init_int_tv<char>(runtime::Tensor& tv)
{
    char lower;
    char upper;
    get_range<char>(lower, upper);
    size_t size = tv.get_element_count();
    uniform_int_distribution<int16_t> dist(static_cast<short>(lower), static_cast<short>(upper));
    vector<char> vec(size);
    for (char& element : vec)
    {
        element = static_cast<char>(dist(s_random_engine));
    }
    tv.write(vec.data(), 0, vec.size() * sizeof(char));
}

template <>
void init_int_tv<int8_t>(runtime::Tensor& tv)
{
    int8_t lower;
    int8_t upper;
    get_range<int8_t>(lower, upper);
    size_t size = tv.get_element_count();
    uniform_int_distribution<int16_t> dist(static_cast<short>(lower), static_cast<short>(upper));
    vector<int8_t> vec(size);
    for (int8_t& element : vec)
    {
        element = static_cast<int8_t>(dist(s_random_engine));
    }
    tv.write(vec.data(), 0, vec.size() * sizeof(int8_t));
}

template <>
void init_int_tv<uint8_t>(runtime::Tensor& tv)
{
    uint8_t lower;
    uint8_t upper;
    get_range<uint8_t>(lower, upper);
    size_t size = tv.get_element_count();
    uniform_int_distribution<int16_t> dist(static_cast<short>(lower), static_cast<short>(upper));
    vector<uint8_t> vec(size);
    for (uint8_t& element : vec)
    {
        element = static_cast<uint8_t>(dist(s_random_engine));
    }
    tv.write(vec.data(), 0, vec.size() * sizeof(uint8_t));
}

template <typename T>
void init_real_tv(runtime::Tensor& tv)
{
    T lower;
    T upper;
    get_range<T>(lower, upper);
    size_t size = tv.get_element_count();
    uniform_real_distribution<T> dist(lower, upper);
    vector<T> vec(size);
    for (T& element : vec)
    {
        element = dist(s_random_engine);
    }
    tv.write(vec.data(), 0, vec.size() * sizeof(T));
}

void random_init(runtime::Tensor& tv)
{
    element::Type et = tv.get_element_type();
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
