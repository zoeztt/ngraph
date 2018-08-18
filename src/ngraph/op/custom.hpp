/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <initializer_list>
#include <memory>
#include <string>

#include "ngraph/op/op.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Custom op
        class Custom : public ngraph::op::Op
        {
        public:
            using execute_t =
                std::function<void(void* user_data,
                                   runtime::Backend* backend,
                                   const std::vector<std::shared_ptr<runtime::TensorView>>& out,
                                   const std::vector<std::shared_ptr<runtime::TensorView>>& args)>;

            /// \brief Constructs a Custom op.
            ///
            /// \param name Custom op's friendly name.
            /// \param args Parameters describing the input tensors.
            Custom(const std::string& name, const NodeVector& args);

            void add_exec(const std::string& backend, execute_t);

            execute_t get_exec(const std::string& backend_name) const;
            void* get_exec_ptr(const std::string& backend_name) const;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

        private:
            std::unordered_map<std::string, execute_t> m_execute_collection;
        };
    }
}
