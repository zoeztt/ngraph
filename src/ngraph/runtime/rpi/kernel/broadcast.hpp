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

#include <Eigen/Dense>
#include <cmath>
#include <omp.h>
#include <utility>

#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace rpi
        {
            namespace kernel
            {
                template <typename T>
                void broadcast_2d(const T* in,
                                  T* out,
                                  const Shape& in_shape,
                                  const Shape& out_shape,
                                  const AxisSet& broadcast_axes)
                {
                    size_t index[2];
                    size_t* out_index =
                        (broadcast_axes.find(0) == broadcast_axes.end() ? &index[0] : &index[1]);
                    for (index[0] = 0; index[0] < out_shape[0]; ++index[0])
                    {
                        for (index[1] = 0; index[1] < out_shape[1]; ++index[1])
                        {
                            out[index[0] * out_shape[1] + index[1]] = in[*out_index];
                        }
                    }
                }

                template <typename T>
                void broadcast(const T* in,
                               T* out,
                               const Shape& in_shape,
                               const Shape& out_shape,
                               const AxisSet& broadcast_axes)
                {
                    // NGRAPH_INFO << "in_shape=" << in_shape << ", out_shape=" << out_shape
                    //             << ",  broadcast_axes(" << broadcast_axes.size()
                    //             << ")=" << join(broadcast_axes);
                    if (in_shape.size() == 0)
                    {
                        for (size_t i = 0; i < shape_size(out_shape); ++i)
                        {
                            out[i] = in[0];
                        }
                    }
                    else if (in_shape.size() == 1 && out_shape.size() == 2)
                    {
                        broadcast_2d<T>(in, out, in_shape, out_shape, broadcast_axes);
                    }
                    else
                    {
                        NGRAPH_INFO << "reference Broadcast";
                        runtime::reference::broadcast<T>(
                            in, out, in_shape, out_shape, broadcast_axes);
                    }
                }
            }
        }
    }
}
