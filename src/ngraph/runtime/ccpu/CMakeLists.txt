# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

set(SRC
    ccpu_backend.cpp
    ccpu_call_frame.cpp
    ccpu_emitter.cpp
    ccpu_external_function.cpp
    ccpu_kernel_emitters.cpp
    ccpu_kernel_utils.cpp
    ccpu_kernels.cpp
    ccpu_layout_descriptor.cpp
    ccpu_tensor_view_wrapper.cpp
    ccpu_tensor_view.cpp
    kernel/eigen_thread_pool.cpp
    kernel/pad.cpp
    kernel/reduce_max.cpp
    kernel/reduce_sum.cpp
    kernel/reshape.cpp
    mkldnn_emitter.cpp
    mkldnn_invoke.cpp
    mkldnn_utils.cpp
    op/batch_dot.cpp
    op/batch_norm_relu.cpp
    op/bounded_relu.cpp
    op/conv_bias.cpp
    op/conv_relu.cpp
    op/convert_layout.cpp
    op/group_conv.cpp
    op/loop_kernel.cpp
    op/lstm.cpp
    op/matmul_bias.cpp
    op/max_pool_with_indices.cpp
    op/rnn.cpp
    op/sigmoid_mul.cpp
    pass/cpu_assignment.cpp
    pass/cpu_collapse_dims.cpp
    pass/cpu_concat_inputs.cpp
    pass/cpu_fusion.cpp
    pass/cpu_layout.cpp
    pass/cpu_loop_kernel_fusion.cpp
    pass/cpu_mat_fusion.cpp
    pass/cpu_post_layout_optimizations.cpp
    pass/cpu_rnn_fusion.cpp
    pass/cpu_workspace_insertion.cpp
)

if (NGRAPH_CCPU_ENABLE)
    set(NGRAPH_CPU_DEBUGINFO_ENABLE 0 CACHE STRING "Enable debuginfo in the CPU backend")

    add_library(ccpu_backend SHARED ${SRC})
    set_target_properties(ccpu_backend PROPERTIES VERSION ${NGRAPH_VERSION})
    set_source_files_properties(ccpu_emitter.cpp PROPERTIES COMPILE_FLAGS "-fpermissive")

    if(NGRAPH_DISTRIBUTED_ENABLE)
        find_package(MPI REQUIRED)
        add_definitions(-DNGRAPH_DISTRIBUTED)
        include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
        link_directories(${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
        target_link_libraries(ccpu_backend PRIVATE ${MPI_CXX_LIBRARIES})
    endif()

    add_dependencies(ccpu_backend ext_mkldnn ext_eigen)
    target_link_libraries(ccpu_backend PUBLIC ngraph libmkldnn libeigen libjson)
    target_link_libraries(ccpu_backend PUBLIC codegen)
    target_include_directories(ccpu_backend SYSTEM PUBLIC libmkldnn)
    set_target_properties(ccpu_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})

    install(TARGETS ccpu_backend LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB})
endif()
