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
                template <typename T, unsigned int IN_RANK, unsigned int OUT_RANK>
                void reshape_in_out(const T* input,
                                    T* output,
                                    const Shape& input_shape,
                                    const AxisVector& input_axis_order,
                                    const Shape& output_shape)
                {
                    NGRAPH_INFO << IN_RANK << " x " << OUT_RANK;
                    Eigen::array<Eigen::Index, OUT_RANK> out_dims;
                    Eigen::array<Eigen::Index, IN_RANK> in_dims;
                    Eigen::array<Eigen::Index, IN_RANK> axis_order;

                    for (int i = 0; i < OUT_RANK; i++)
                    {
                        out_dims[i] = output_shape[i];
                    }

                    for (int i = 0; i < IN_RANK; i++)
                    {
                        in_dims[i] = input_shape[i];
                        axis_order[i] = input_axis_order[i];
                    }

                    Eigen::TensorMap<Eigen::Tensor<T, OUT_RANK, Eigen::RowMajor>> out(output,
                                                                                      out_dims);
                    Eigen::TensorMap<Eigen::Tensor<const T, IN_RANK, Eigen::RowMajor>> in(input,
                                                                                          in_dims);

                    out = in.shuffle(axis_order).reshape(out_dims);
                }

                template <typename T, unsigned int IN_RANK>
                void reshape_in(const T* input,
                                T* output,
                                const Shape& input_shape,
                                const AxisVector& input_axis_order,
                                const Shape& output_shape)
                {
                    NGRAPH_INFO << IN_RANK;
                    switch (output_shape.size())
                    {
                    // case 0:
                    //     reshape_in_out<T, IN_RANK, 0>(
                    //         input, output, input_shape, input_axis_order, output_shape);
                    //     break;
                    case 1:
                        reshape_in_out<T, IN_RANK, 1>(
                            input, output, input_shape, input_axis_order, output_shape);
                        break;
                    case 2:
                        reshape_in_out<T, IN_RANK, 2>(
                            input, output, input_shape, input_axis_order, output_shape);
                        break;
                    case 3:
                        reshape_in_out<T, IN_RANK, 3>(
                            input, output, input_shape, input_axis_order, output_shape);
                        break;
                    case 4:
                        reshape_in_out<T, IN_RANK, 4>(
                            input, output, input_shape, input_axis_order, output_shape);
                        break;
                    case 5:
                        reshape_in_out<T, IN_RANK, 5>(
                            input, output, input_shape, input_axis_order, output_shape);
                        break;
                    default:
                        reference::reshape(static_cast<const T*>(input),
                                           static_cast<T*>(output),
                                           input_shape,
                                           input_axis_order,
                                           output_shape);
                        break;
                    }
                }

                template <typename T>
                void reshape(const T* input,
                             T* output,
                             const Shape& input_shape,
                             const AxisVector& input_axis_order,
                             const Shape& output_shape)
                {
                    NGRAPH_INFO << input_shape.size() << " x " << output_shape.size();

                    switch (input_shape.size())
                    {
                    // case 0:
                    //     reshape_in<T, 0>(
                    //         input, output, input_shape, input_axis_order, output_shape);
                    //     break;
                    case 1:
                        reshape_in<T, 1>(
                            input, output, input_shape, input_axis_order, output_shape);
                        break;
                    case 2:
                        reshape_in<T, 2>(
                            input, output, input_shape, input_axis_order, output_shape);
                        break;
                    case 3:
                        reshape_in<T, 3>(
                            input, output, input_shape, input_axis_order, output_shape);
                        break;
                    case 4:
                        reshape_in<T, 4>(
                            input, output, input_shape, input_axis_order, output_shape);
                        break;
                    case 5:
                        reshape_in<T, 5>(
                            input, output, input_shape, input_axis_order, output_shape);
                        break;
                    default:
                        reference::reshape(
                            input, output, input_shape, input_axis_order, output_shape);
                        break;
                    }
                }
            }
        }
    }
}
