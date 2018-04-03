/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <immintrin.h>

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                void ewadd_float32(float* input0,
                                   float* input1,
                                   float* output,
                                   size_t N)
                {
                    constexpr int UF = 8;
                    constexpr auto BS = 16*UF*256;
                    auto NB = N / BS;

                    #pragma omp parallel for
                    for (size_t i = 0; i < NB; i++)
                    {
                        size_t j = BS*i;
                        size_t E = j + BS;
                        for (; j < E; j += UF*16)
                        {
                            for (size_t k = 0; k < UF; k++)
                            {
                                __m512 i0, i1, out;
                                //i0 = _mm512_load_ps(input0 + j + k*16);
                                //i1 = _mm512_load_ps(input1 + j + k*16);
                                i0 = _mm512_stream_load_si512(input0 + j + k*16);
                                i1 = _mm512_stream_load_si512(input1 + j + k*16);
                                out = _mm512_add_ps(i0, i1);
                                //_mm512_store_ps(output + j + k*16, out);
                                _mm512_stream_ps(output + j + k*16, out);
                            }
                        }
                    }

                    auto RSI = NB*BS;
                    auto RS = N - NB*BS;
                    auto B = RS / 16;
                    for (size_t i = 0; i < B; i++)
                    {
                        __m512 i0, i1, out;
                        i0 = _mm512_load_ps(input0 + RSI + i*16);
                        i1 = _mm512_load_ps(input1 + RSI + i*16);
                        out = _mm512_add_ps(i0, i1);
                        _mm512_store_ps(output + RSI + i*16, out);
                    }
                    RSI += B*16;
                    for (; RSI < N; RSI++)
                    {
                        output[RSI] = input0[RSI] + input1[RSI];
                    }
                }
            }
        }
    }
}
