
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

#------------------------------------------------------------------------------
# Download pybind11
#------------------------------------------------------------------------------

SET(JSON_GIT_REPO_URL https://github.com/jagerman/pybind11.git)
SET(JSON_GIT_LABEL v1.8.1)

# The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
if (${CMAKE_VERSION} VERSION_LESS 3.2)
    ExternalProject_Add(
        ext_pybind11
        PREFIX pybind11
        GIT_REPOSITORY ${JSON_GIT_REPO_URL}
        GIT_TAG ${JSON_GIT_LABEL}
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        EXCLUDE_FROM_ALL TRUE
        )
else()
    ExternalProject_Add(
        ext_pybind11
        PREFIX pybind11
        GIT_REPOSITORY ${JSON_GIT_REPO_URL}
        GIT_TAG ${JSON_GIT_LABEL}
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        EXCLUDE_FROM_ALL TRUE
    )
endif()

#------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_pybind11 SOURCE_DIR)
add_library(libpybind11 INTERFACE)
target_include_directories(libpybind11 SYSTEM INTERFACE ${SOURCE_DIR}/include)
add_dependencies(libpybind11 ext_pybind11)
