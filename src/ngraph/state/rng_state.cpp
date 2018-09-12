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

#include <random>

#include "except.hpp"
#include "rng_state.hpp"

using namespace std;
using namespace ngraph;

void ngraph::RNGState::activate()
{
    //reentrant?
    if (!is_active())
    {
        std::random_device rd;
        m_seed = rd();
        set_active(true);
    }
}

void ngraph::RNGState::deactivate()
{
    if (!is_active())
    {
        throw ngraph_error("State wasn't activated");
    }

    set_active(false);
}
