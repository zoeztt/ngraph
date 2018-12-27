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
                    NGRAPH_INFO;
                    size_t size[2];
                    size_t in_index[2];
                    size_t* map_index[2];
                    for (size_t i = 0; i < 2; i++)
                    {
                        size[i] = in_shape[in_axis_order[i]];
                        map_index[i] = &in_index[in_axis_order[i]];
                    }
                    for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
                    {
                        for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1])
                        {
                            NGRAPH_INFO << *map_index[0] << ", " << *map_index[1] << " -> "
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
                    NGRAPH_INFO;
                    size_t size[3];
                    size_t in_index[3];
                    size_t* map_index[3];
                    for (size_t i = 0; i < 3; i++)
                    {
                        // 2, 3, 4 order 1, 2, 0 -> 3, 4, 2
                        size[i] = in_shape[in_axis_order[i]];
                        map_index[in_axis_order[i]] = &in_index[i];
                        // size[i] = in_shape[i];
                        // map_index[i] = &in_index[i];
                    }
                    NGRAPH_INFO << "size = " << size[0] << ", " << size[1] << ", " << size[2];
                    // NGRAPH_INFO << "map_index[" << i << "] = " << map_index[0] << ", "
                    //             << map_index[1] << ", " << map_index[2];
                    for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0]) // axis 1
                    {
                        for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1]) // axis 2
                        {
                            for (in_index[2] = 0; in_index[2] < size[2]; ++in_index[2]) // axis 0
                            {
                                // NGRAPH_INFO << "in_index " << in_index[0] << ", " << in_index[1]
                                //             << ", " << in_index[2];
                                NGRAPH_INFO << *map_index[0] << ", " << *map_index[1] << ", "
                                            << *map_index[2] << " -> "
                                            << (*map_index[0] * size[1] * size[2] +
                                                *map_index[1] * size[2] + *map_index[2]);
                                *out++ = in[*map_index[0] * in_shape[1] * in_shape[2] +
                                            *map_index[1] * in_shape[2] + *map_index[2]];
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
                    case 2: reshape_in2<T>(in, out, in_shape, in_axis_order, out_shape); break;
                    case 3:
                        reshape_in3<T>(in, out, in_shape, in_axis_order, out_shape);
                        break;
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
