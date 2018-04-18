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

# Enable ExternalProject CMake module
include(ExternalProject)

#----------------------------------------------------------------------------------------------------------
# Download and install GoogleTest ...
#----------------------------------------------------------------------------------------------------------

SET(PYBIND11_GIT_REPO_URL https://github.com/pybind/pybind11.git)
SET(PYBIND11_GIT_LABEL v2.2.2)
set(PYBIND11_HEADER_DIR "")
set(PYBIND11_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/pybind11)

# The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
if (${CMAKE_VERSION} VERSION_LESS 3.2)
    ExternalProject_Add(
        pybind11
        GIT_REPOSITORY ${PYBIND11_GIT_REPO_URL}
        GIT_TAG ${PYBIND11_GIT_LABEL}
        PREFIX ${PYBIND11_PREFIX}
        # Disable install step
        UPDATE_COMMAND ""
        INSTALL_COMMAND ""
        CMAKE_ARGS -DPYBIND11_INSTALL=${PYBIND11_HEADER_DIR}
                   -DPYBIND11_TEST=FALSE
                   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                   -DCMAKE_CXX_FLAGS="-fPIC"
        )
else()
    ExternalProject_Add(
        pybind11
        GIT_REPOSITORY ${PYBIND11_GIT_REPO_URL}
        GIT_TAG ${PYBIND11_GIT_LABEL}
        PREFIX ${PYBIND11_PREFIX}
        # Disable install step
        UPDATE_COMMAND ""
        INSTALL_COMMAND ""
        CMAKE_ARGS -DPYBIND11_INSTALL=${PYBIND11_HEADER_DIR}
                   -DPYBIND11_TEST=FALSE
                   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                   -DCMAKE_CXX_FLAGS="-fPIC"
        )
endif()

#----------------------------------------------------------------------------------------------------------

get_filename_component(
    PYBIND11_INCLUDE_DIR
    "${EXTERNAL_PROJECTS_ROOT}/pybind11/src/googletest/include"
    ABSOLUTE)
set(PYBIND11_INCLUDE_DIR "${PYBIND11_INCLUDE_DIR}" PARENT_SCOPE)

# Create a libpybind11 target to be used as a dependency by test programs
add_library(libpybind11 IMPORTED STATIC GLOBAL)
add_dependencies(libpybind11 pybind11)
