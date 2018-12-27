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

#pragma once

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "ngraph/axis_vector.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace rpi
        {
            namespace kernel
            {
                template <typename T>
                void reshape_in2(const T* in,
                                 T* out,
                                 const Shape& in_shape,
                                 const AxisVector& in_axis_order,
                                 const Shape& out_shape)
                {
                    size_t size[2];
                    size_t in_index[2];
                    size_t* map_index[2];
                    size_t* t[2] = {&in_index[0], &in_index[1]};
                    for (size_t i = 0; i < 2; i++)
                    {
                        size[i] = in_shape[in_axis_order[i]];
                        map_index[i] = &in_index[in_axis_order[i]];
                    }
                    NGRAPH_INFO << size[0] << ", " << size[1];
                    for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
                    {
                        for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1])
                        {
                            NGRAPH_INFO << in_index[0] << ", " << in_index[1] << "         "
                                        << *map_index[0] << ", " << *map_index[1] << " -> "
                                        << (*map_index[0] * in_shape[1] + *map_index[1]);
                            *out++ = in[*map_index[0] * in_shape[1] + *map_index[1]];
                        }
                    }
                }

                template <typename T>
                void reshape_in3(const T* in,
                                 T* out,
                                 const Shape& in_shape,
                                 const AxisVector& in_axis_order,
                                 const Shape& out_shape)
                {
                    size_t o0_size;
                    size_t o1_size;
                    size_t o2_size;
                    size_t i0;
                    size_t i1;
                    size_t i2;
                    size_t o0;
                    size_t o1;
                    size_t o2;
                    size_t* index[3] = {&o0_size, &o1_size, &o2_size};
                    for (size_t i = 0; i < 3; i++)
                    {
                        *index[i] = in_shape[in_axis_order[i]];
                    }
                    NGRAPH_INFO << o0_size << ", " << o1_size << ", " << o2_size;
                    for (i0 = 0; i0 < in_shape[0]; ++i0)
                    {
                        for (i1 = 0; i1 < in_shape[1]; ++i1)
                        {
                            for (i2 = 0; i2 < in_shape[2]; ++i2)
                            {
                                out[o0 * out_shape[1] * out_shape[2] + o1 * out_shape[2] + o2] =
                                    in[i0 * in_shape[1] * in_shape[2] + i1 * in_shape[2] + i2];
                            }
                        }
                    }
                }

                template <typename T>
                void reshape(const T* in,
                             T* out,
                             const Shape& in_shape,
                             const AxisVector& in_axis_order,
                             const Shape& out_shape)
                {
                    NGRAPH_INFO << in_shape << " -> " << out_shape << " order "
                                << join(in_axis_order);
                    NGRAPH_INFO << shape_size(in_shape) << " size " << shape_size(out_shape);

                    switch (in_shape.size())
                    {
                    // case 0: reshape_in0<T>(in, out, in_shape, in_axis_order, out_shape); break;
                    // case 1: reshape_in1<T>(in, out, in_shape, in_axis_order, out_shape); break;
                    case 2:
                        reshape_in2<T>(in, out, in_shape, in_axis_order, out_shape);
                        break;
                    // case 3: reshape_in3<T>(in, out, in_shape, in_axis_order, out_shape); break;
                    // case 4: reshape_in4<T>(in, out, in_shape, in_axis_order, out_shape); break;
                    // case 5: reshape_in<5T>(in, out, in_shape, in_axis_order, out_shape); break;
                    default:
                        NGRAPH_INFO << "reference::reshape";
                        reference::reshape(in, out, in_shape, in_axis_order, out_shape);
                        break;
                    }
                }
            }
        }
    }
}
