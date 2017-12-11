// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <memory>

#include <iostream>

#include "ngraph/except.hpp"
#include "ngraph/log.hpp"
#include "ngraph/types/type.hpp"
#include "ngraph/util.hpp"



using namespace std;
using namespace ngraph;

bool TensorViewType::operator==(const ValueType& that) const
{
	
	std::cout << "Comparing TensorViewType\n";
    auto that_tvt = dynamic_cast<const TensorViewType*>(&that);
    if (nullptr == that_tvt)
    {
		std::cout << "nullptr == that_tvt\n";
        return false;
    }


    if (that_tvt->get_element_type() != m_element_type)
    {
		std::cout << "Types are not equal this_type = " << m_element_type.c_type_string() << " , that_type = " << that_tvt->get_element_type().c_type_string() << std::endl;
        return false;
    }
    if (that_tvt->get_shape() != m_shape)
    {
		std::cout << "shapes aren't equal\n";
        return false;
    }
    return true;
}

void TensorViewType::collect_tensor_views(
    std::vector<std::shared_ptr<const TensorViewType>>& views) const
{
    views.push_back(shared_from_this());
}

bool TupleType::operator==(const ValueType& that) const
{
    auto that_tvt = dynamic_cast<const TupleType*>(&that);
    if (nullptr == that_tvt)
    {
		std::cout << "that_tvt is null\n";
        return false;
    }

	std::cout << "this size = " << get_element_types().size() << " , that size = " << that_tvt->get_element_types().size() << std::endl;
	if (get_element_types().size() != that_tvt->get_element_types().size())
	{
		return false;
	}
	for (size_t i = 0; i < get_element_types().size(); i++) 
	{
		std::cout << "comparing " << i << " th element\n";
		if (*get_element_types().at(i) != *that_tvt->get_element_types().at(i))
		{
			return false;
		}
		
	}
	return true;
}

void TupleType::collect_tensor_views(
    std::vector<std::shared_ptr<const TensorViewType>>& views) const
{
    for (auto elt : m_element_types)
    {
        elt->collect_tensor_views(views);
    }
}

const Shape& TupleType::get_shape() const
{
    throw ngraph_error("get_shape() called on Tuple");
}

const element::Type& TupleType::get_element_type() const
{
    throw ngraph_error("get_element_type() called on Tuple");
}

std::ostream& ngraph::operator<<(std::ostream& out, const ValueType& obj)
{
    out << "ValueType()";
    return out;
}

std::ostream& ngraph::operator<<(std::ostream& out, const TensorViewType& obj)
{
    out << "TensorViewType(" << obj.m_element_type << ", {" << join(obj.m_shape) << "})";
    return out;
}

std::ostream& ngraph::operator<<(std::ostream& out, const TupleType& obj)
{
    out << "TupleType()";
    return out;
}
