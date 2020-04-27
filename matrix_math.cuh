/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef __MATRIX_MATH_HXX__
#define __MATRIX_MATH_HXX__

template <int M, int K>
__device__ __inline__ void loadWeights(float weights_local[K], float* weights_remote, int layer, int row, int lda=M) {

    if (row >= M) return;

#pragma unroll
    for (int i=0; i<K; i++) {
        weights_local[i] = weights_remote[lda*K*layer + lda*i + row];
    }

}

template <int K, int K_UNROLL, int TILE_N>
__device__ void GEMM(float weights[K], float activations[TILE_N][K], float accum[TILE_N]) {

    float accum_unrolled[TILE_N][K_UNROLL];

#pragma unroll
    for (int n=0; n<TILE_N; n++) {
#pragma unroll
        for (int u=0; u<K_UNROLL; u++) {
            accum_unrolled[n][u] = 0.f;
        }
    }

#pragma unroll
    for (int i=0; i<K; i += K_UNROLL) {
#pragma unroll
        for (int n=0; n<TILE_N; n++) {
#pragma unroll
            for (int u=0; u<K_UNROLL; u++) {
                accum_unrolled[n][u] += weights[i+u]*activations[n][i+u];
            }
        }
    }

#pragma unroll
    for (int n=0; n<TILE_N; n++) {
#pragma unroll
        for (int u=1; u<K_UNROLL; u++) {
            accum_unrolled[n][0] += accum_unrolled[n][u];
        }
    }

#pragma unroll
    for (int n=0; n<TILE_N; n++) {
        accum[n] = accum_unrolled[n][0];
    }

}

#endif
