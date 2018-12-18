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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/op/parameter.hpp"
#include "ngraph/type/element_type.hpp"
#include "pyngraph/types/element_type.hpp"

namespace py = pybind11;

void regclass_pyngraph_Type(py::module m)
{
    py::class_<ngraph::Type, std::shared_ptr<ngraph::Type>> type(m, "Type");
    type.doc() = "ngraph.impl.Type wraps ngraph::Type";
    type.attr("boolean") = ngraph::boolean;
    type.attr("f32") = ngraph::f32;
    type.attr("f64") = ngraph::f64;
    type.attr("i8") = ngraph::i8;
    type.attr("i16") = ngraph::i16;
    type.attr("i32") = ngraph::i32;
    type.attr("i64") = ngraph::i64;
    type.attr("u8") = ngraph::u8;
    type.attr("u16") = ngraph::u16;
    type.attr("u32") = ngraph::u32;
    type.attr("u64") = ngraph::u64;

    type.def("__repr__", [](const ngraph::Type& self) {
        std::string bitwidth = std::to_string(self.bitwidth());
        if (self.is_signed())
        {
            return "<Type: '" + self.c_type_string() + bitwidth + "'>";
        }
        return "<Type: 'u" + self.c_type_string() + bitwidth + "'>";
    });

    type.def("__eq__",
             [](const ngraph::Type& a, const ngraph::Type& b) { return a == b; },
             py::is_operator());

    type.def_property_readonly("bitwidth", &ngraph::Type::bitwidth);
    type.def_property_readonly("is_real", &ngraph::Type::is_real);
}
