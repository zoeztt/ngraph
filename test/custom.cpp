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

#include <dlfcn.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/log.hpp"
#include "ngraph/op/custom.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/ccpu/ccpu_backend.hpp"
#include "ngraph/runtime/ccpu/ccpu_tensor_view.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

class AbcOp : public op::Custom
{
public:
    AbcOp(const std::shared_ptr<Node>& arg0,
          const std::shared_ptr<Node>& arg1,
          const std::shared_ptr<Node>& arg2)
        : Custom("ABC", {arg0, arg1, arg2})
    {
        // add_exec(
        //     "INTERPRETER",
        //     bind(&AbcOp::execute, this, placeholders::_1, placeholders::_2, placeholders::_3));
        // add_exec(
        //     "CCPU",
        //     bind(&AbcOp::execute, this, placeholders::_1, placeholders::_2, placeholders::_3));
        add_exec("INTERPRETER", execute);
        add_exec("CCPU", execute);

        if (arg0->get_element_type() != arg1->get_element_type() ||
            arg0->get_element_type() != arg2->get_element_type())
        {
            throw ngraph_error("Arguments must have the same tensor view element type");
        }

        if (arg0->get_shape() != arg1->get_shape() || arg0->get_shape() != arg2->get_shape())
        {
            throw ngraph_error("Arguments must have the same tensor view shape");
        }

        const element::Type& element_type = get_input_element_type(0);
        const Shape& shape = get_input_shape(0);
        set_output_type(0, element_type, shape);
    }

private:
    static void execute(void* user_data,
                        runtime::Backend* backend,
                        const std::vector<std::shared_ptr<runtime::TensorView>>& out,
                        const std::vector<std::shared_ptr<runtime::TensorView>>& args)
    {
        if (dynamic_cast<runtime::interpreter::INTBackend*>(backend))
        {
            const float* arg0 =
                dynamic_pointer_cast<runtime::HostTensorView>(args[0])->get_data_ptr<float>();
            const float* arg1 =
                dynamic_pointer_cast<runtime::HostTensorView>(args[1])->get_data_ptr<float>();
            const float* arg2 =
                dynamic_pointer_cast<runtime::HostTensorView>(args[2])->get_data_ptr<float>();
            float* out0 =
                dynamic_pointer_cast<runtime::HostTensorView>(out[0])->get_data_ptr<float>();
            size_t size = out[0]->get_element_count();
            for (size_t i = 0; i < size; i++)
            {
                out0[i] = (arg0[i] + arg1[i]) * arg2[i];
            }
        }
        else if (dynamic_cast<runtime::ccpu::CCPUBackend*>(backend))
        {
            const float* arg0 = reinterpret_cast<float*>(
                dynamic_pointer_cast<runtime::ccpu::CCPUTensorView>(args[0])->get_data_ptr());
            const float* arg1 = reinterpret_cast<float*>(
                dynamic_pointer_cast<runtime::ccpu::CCPUTensorView>(args[1])->get_data_ptr());
            const float* arg2 = reinterpret_cast<float*>(
                dynamic_pointer_cast<runtime::ccpu::CCPUTensorView>(args[2])->get_data_ptr());
            float* out0 = reinterpret_cast<float*>(
                dynamic_pointer_cast<runtime::ccpu::CCPUTensorView>(out[0])->get_data_ptr());
            size_t size = out[0]->get_element_count();
            for (size_t i = 0; i < size; i++)
            {
                out0[i] = (arg0[i] + arg1[i]) * arg2[i];
            }
        }
    }

    shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override
    {
        if (new_args.size() != 3)
        {
            throw ngraph_error("Incorrect number of new arguments for AbcOp");
        }
        return make_shared<AbcOp>(new_args.at(0), new_args.at(1), new_args.at(2));
    }
};

class unsupported_op : public op::Custom
{
public:
    unsupported_op()
        : Custom("Unsupported", {})
    {
        constructor_validate_and_infer_types();
    }

private:
    shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override
    {
        return make_shared<unsupported_op>();
    }
};

TEST(custom_op, abc)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto abc = make_shared<AbcOp>(A, B, C);
    auto f = make_shared<Function>(abc, op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(f, {result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(f, {result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());
}

TEST(custom_op, unsupported)
{
    auto unsupported = make_shared<unsupported_op>();
    auto x = op::ParameterVector{};
    auto f = make_shared<Function>(unsupported, x);

    auto backend = runtime::Backend::create("INTERPRETER");
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, {});

    EXPECT_ANY_THROW(backend->call_with_validate(f, {result}, {}));
}

class SoOp : public op::Custom
{
public:
    SoOp(const std::shared_ptr<Node>& arg0,
         const std::shared_ptr<Node>& arg1,
         const std::shared_ptr<Node>& arg2)
        : Custom("SO", {arg0, arg1, arg2})
    {
        add_exec("INTERPRETER",
                      bind(&SoOp::execute,
                           this,
                           placeholders::_1,
                           placeholders::_2,
                           placeholders::_3,
                           placeholders::_4));

        if (arg0->get_element_type() != arg1->get_element_type() ||
            arg0->get_element_type() != arg2->get_element_type())
        {
            throw ngraph_error("Arguments must have the same tensor view element type");
        }

        if (arg0->get_shape() != arg1->get_shape() || arg0->get_shape() != arg2->get_shape())
        {
            throw ngraph_error("Arguments must have the same tensor view shape");
        }

        const element::Type& element_type = get_input_element_type(0);
        const Shape& shape = get_input_shape(0);
        set_output_type(0, element_type, shape);

        void* handle = dlopen("./libcustom_op_library.so", RTLD_NOW | RTLD_GLOBAL);
        if (!handle)
        {
            throw runtime_error("library not found");
        }
        m_execute = reinterpret_cast<void (*)(
            void*,
            runtime::Backend* backend,
            const std::vector<std::shared_ptr<runtime::TensorView>>& out,
            const std::vector<std::shared_ptr<runtime::TensorView>>& args)>(
            dlsym(handle, "execute"));
        if (!m_execute)
        {
            throw runtime_error("test library not found");
        }
    }

private:
    void execute(void* user_data,
                 runtime::Backend* backend,
                 const std::vector<std::shared_ptr<runtime::TensorView>>& out,
                 const std::vector<std::shared_ptr<runtime::TensorView>>& args) const
    {
        m_execute(nullptr, backend, out, args);
    }

    shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override
    {
        if (new_args.size() != 3)
        {
            throw ngraph_error("Incorrect number of new arguments for AbcOp");
        }
        return make_shared<AbcOp>(new_args.at(0), new_args.at(1), new_args.at(2));
    }

    execute_t m_execute;
};

TEST(custom_op, shared_lib)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto abc = make_shared<SoOp>(A, B, C);
    auto f = make_shared<Function>(abc, op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(f, {result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(f, {result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());
}

TEST(custom_op, codegen)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto abc = make_shared<AbcOp>(A, B, C);
    auto f = make_shared<Function>(abc, op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("CCPU");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(f, {result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(f, {result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());
}
