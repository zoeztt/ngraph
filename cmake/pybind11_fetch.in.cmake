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

cmake_minimum_required(VERSION 3.1)

include(ExternalProject)

project(pybind11-fetch NONE)

#------------------------------------------------------------------------------
# Download pybind11
#------------------------------------------------------------------------------

SET(PYBIND11_GIT_REPO_URL https://github.com/jagerman/pybind11.git)
SET(PYBIND11_GIT_LABEL v1.8.1)

ExternalProject_Add(
    ext_pybind11
    GIT_REPOSITORY ${PYBIND11_GIT_REPO_URL}
    GIT_TAG ${PYBIND11_GIT_LABEL}
    # SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pybind11/pybind11-src"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
)
