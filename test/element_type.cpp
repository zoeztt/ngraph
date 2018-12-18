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

#include <map>

#include "gtest/gtest.h"

#include "ngraph/type/element_type.hpp"

using namespace ngraph;

TEST(element_type, from)
{
    EXPECT_EQ(from<char>(), boolean);
    EXPECT_EQ(from<bool>(), boolean);
    EXPECT_EQ(from<float>(), f32);
    EXPECT_EQ(from<double>(), f64);
    EXPECT_EQ(from<int8_t>(), i8);
    EXPECT_EQ(from<int16_t>(), i16);
    EXPECT_EQ(from<int32_t>(), i32);
    EXPECT_EQ(from<int64_t>(), i64);
    EXPECT_EQ(from<uint8_t>(), u8);
    EXPECT_EQ(from<uint16_t>(), u16);
    EXPECT_EQ(from<uint32_t>(), u32);
    EXPECT_EQ(from<uint64_t>(), u64);
}

TEST(element_type, mapable)
{
    std::map<Type, std::string> test_map;

    test_map.insert({f32, "float"});
}

TEST(element_type, size)
{
    {
        Type t1{1, false, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        Type t1{2, false, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        Type t1{3, false, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        Type t1{4, false, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        Type t1{5, false, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        Type t1{6, false, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        Type t1{7, false, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        Type t1{8, false, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        Type t1{9, false, false, false, ""};
        EXPECT_EQ(2, t1.size());
    }
}

TEST(element_type, merge_both_dynamic)
{
    Type t;
    ASSERT_TRUE(Type::merge(t, dynamic, dynamic));
    ASSERT_TRUE(t.is_dynamic());
}

TEST(element_type, merge_left_dynamic)
{
    Type t;
    ASSERT_TRUE(Type::merge(t, dynamic, u64));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, u64);
}

TEST(element_type, merge_right_dynamic)
{
    Type t;
    ASSERT_TRUE(Type::merge(t, i16, dynamic));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, i16);
}

TEST(element_type, merge_both_static_equal)
{
    Type t;
    ASSERT_TRUE(Type::merge(t, f64, f64));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, f64);
}

TEST(element_type, merge_both_static_unequal)
{
    Type t = f32;
    ASSERT_FALSE(Type::merge(t, i8, i16));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, f32);
}
