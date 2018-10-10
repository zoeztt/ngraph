# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

get_target_property(PYBIND11_INCLUDE_DIR libpybind11 INTERFACE_INCLUDE_DIRECTORIES)
set(NGRAPH_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

SET(WRAPPER_SRC
    pyngraph/function.cpp
    pyngraph/serializer.cpp
    pyngraph/node.cpp
    pyngraph/node_vector.cpp
    pyngraph/onnx_import/onnx_import.cpp
    pyngraph/shape.cpp
    pyngraph/strides.cpp
    pyngraph/coordinate_diff.cpp
    pyngraph/axis_set.cpp
    pyngraph/axis_vector.cpp
    pyngraph/coordinate.cpp
    pyngraph/pyngraph.cpp
    pyngraph/util.cpp
    pyngraph/ops/util/arithmetic_reduction.cpp
    pyngraph/ops/util/binary_elementwise_comparison.cpp
    pyngraph/ops/util/op_annotations.cpp
    pyngraph/ops/util/binary_elementwise_arithmetic.cpp
    pyngraph/ops/util/binary_elementwise_logical.cpp
    pyngraph/ops/util/regmodule_pyngraph_op_util.cpp
    pyngraph/ops/util/unary_elementwise_arithmetic.cpp
    pyngraph/ops/util/index_reduction.cpp
    pyngraph/ops/abs.cpp
    pyngraph/ops/acos.cpp
    pyngraph/ops/add.cpp
    pyngraph/ops/and.cpp
    pyngraph/ops/argmax.cpp
    pyngraph/ops/argmin.cpp
    pyngraph/ops/asin.cpp
    pyngraph/ops/atan.cpp
    pyngraph/ops/avg_pool.cpp
    pyngraph/ops/broadcast.cpp
    pyngraph/ops/concat.cpp
    pyngraph/ops/constant.cpp
    pyngraph/ops/convert.cpp
    pyngraph/ops/convolution.cpp
    pyngraph/ops/cos.cpp
    pyngraph/ops/cosh.cpp
    pyngraph/ops/ceiling.cpp
    pyngraph/ops/divide.cpp
    pyngraph/ops/dot.cpp
    pyngraph/ops/equal.cpp
    pyngraph/ops/exp.cpp
    pyngraph/ops/floor.cpp
    pyngraph/ops/greater.cpp
    pyngraph/ops/greater_eq.cpp
    pyngraph/ops/less.cpp
    pyngraph/ops/less_eq.cpp
    pyngraph/ops/log.cpp
    pyngraph/ops/lrn.cpp
    pyngraph/ops/maximum.cpp
    pyngraph/ops/max.cpp
    pyngraph/ops/product.cpp
    pyngraph/ops/max_pool.cpp
    pyngraph/ops/minimum.cpp
    pyngraph/ops/multiply.cpp
    pyngraph/ops/negative.cpp
    pyngraph/ops/not.cpp
    pyngraph/ops/not_equal.cpp
    pyngraph/ops/op.cpp
    pyngraph/ops/one_hot.cpp
    pyngraph/ops/or.cpp
    pyngraph/ops/pad.cpp
    pyngraph/ops/parameter.cpp
    pyngraph/ops/parameter_vector.cpp
    pyngraph/ops/power.cpp
    pyngraph/ops/reduce.cpp
    pyngraph/ops/regmodule_pyngraph_op.cpp
    pyngraph/ops/relu.cpp
    pyngraph/ops/replace_slice.cpp
    pyngraph/ops/reshape.cpp
    pyngraph/ops/reverse.cpp
    pyngraph/ops/select.cpp
    pyngraph/ops/sign.cpp
    pyngraph/ops/sin.cpp
    pyngraph/ops/sinh.cpp
    pyngraph/ops/slice.cpp
    pyngraph/ops/sqrt.cpp
    pyngraph/ops/subtract.cpp
    pyngraph/ops/sum.cpp
    pyngraph/ops/tan.cpp
    pyngraph/ops/tanh.cpp
    pyngraph/ops/topk.cpp
    pyngraph/ops/allreduce.cpp
    pyngraph/ops/function_call.cpp
    pyngraph/ops/get_output_element.cpp
    pyngraph/ops/min.cpp
    pyngraph/ops/batch_norm.cpp
    pyngraph/ops/softmax.cpp
    pyngraph/runtime/backend.cpp
    pyngraph/runtime/regmodule_pyngraph_runtime.cpp
    pyngraph/runtime/tensor.cpp
    pyngraph/passes/manager.cpp
    pyngraph/passes/regmodule_pyngraph_passes.cpp
    pyngraph/types/element_type.cpp
    pyngraph/types/regmodule_pyngraph_types.cpp
)

foreach(FILE ${WRAPPER_SRC})
    set(TMP ${TMP} "'${FILE}'")
endforeach()
string(REPLACE ";" ",\n           " WRAPPER_SRC_LIST "${TMP}")

message(STATUS "******************** PYBIND11_INCLUDE_DIR ${PYBIND11_INCLUDE_DIR}")
message(STATUS "******************** NGRAPH_INSTALL_PREFIX ${NGRAPH_INSTALL_PREFIX}")
message(STATUS "******************** WRAPPER_SRC ${WRAPPER_SRC_LIST}")

configure_file(setup.in.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

# add_custom_command()
