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

#include "random.hpp"

using namespace ngraph;
using namespace std;

namespace ngraph
{
    namespace test
    {
        template <>
        void init_int_tv<char>(runtime::Tensor* tv, char min, char max)
        {
            size_t size = tv->get_element_count();
            uniform_int_distribution<int16_t> dist(static_cast<short>(min),
                                                   static_cast<short>(max));
            vector<char> vec(size);
            for (char& element : vec)
            {
                element = static_cast<char>(dist(s_random_engine));
            }
            tv->write(vec.data(), 0, vec.size() * sizeof(char));
        }

        template <>
        void init_int_tv<int8_t>(runtime::Tensor* tv, int8_t min, int8_t max)
        {
            size_t size = tv->get_element_count();
            uniform_int_distribution<int16_t> dist(static_cast<short>(min),
                                                   static_cast<short>(max));
            vector<int8_t> vec(size);
            for (int8_t& element : vec)
            {
                element = static_cast<int8_t>(dist(s_random_engine));
            }
            tv->write(vec.data(), 0, vec.size() * sizeof(int8_t));
        }

        template <>
        void init_int_tv<uint8_t>(runtime::Tensor* tv, uint8_t min, uint8_t max)
        {
            size_t size = tv->get_element_count();
            uniform_int_distribution<int16_t> dist(static_cast<short>(min),
                                                   static_cast<short>(max));
            vector<uint8_t> vec(size);
            for (uint8_t& element : vec)
            {
                element = static_cast<uint8_t>(dist(s_random_engine));
            }
            tv->write(vec.data(), 0, vec.size() * sizeof(uint8_t));
        }
    }
}

void test::random_init(runtime::Tensor* tv)
{
    element::Type et = tv->get_element_type();
    if (et == element::boolean)
    {
        init_int_tv<char>(tv, 0, 1);
    }
    else if (et == element::f32)
    {
        init_real_tv<float>(tv, -1, 1);
    }
    else if (et == element::f64)
    {
        init_real_tv<double>(tv, -1, 1);
    }
    else if (et == element::i8)
    {
        init_int_tv<int8_t>(tv, -1, 1);
    }
    else if (et == element::i16)
    {
        init_int_tv<int16_t>(tv, -1, 1);
    }
    else if (et == element::i32)
    {
        init_int_tv<int32_t>(tv, 0, 1);
    }
    else if (et == element::i64)
    {
        init_int_tv<int64_t>(tv, -1, 1);
    }
    else if (et == element::u8)
    {
        init_int_tv<uint8_t>(tv, 0, 1);
    }
    else if (et == element::u16)
    {
        init_int_tv<uint16_t>(tv, 0, 1);
    }
    else if (et == element::u32)
    {
        init_int_tv<uint32_t>(tv, 0, 1);
    }
    else if (et == element::u64)
    {
        init_int_tv<uint64_t>(tv, 0, 1);
    }
    else
    {
        throw runtime_error("unsupported type");
    }
}
