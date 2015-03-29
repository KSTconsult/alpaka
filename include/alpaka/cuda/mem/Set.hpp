/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/cuda/mem/Space.hpp>    // SpaceCuda
#include <alpaka/cuda/Stream.hpp>       // StreamCuda
#include <alpaka/cuda/Common.hpp>

#include <alpaka/core/BasicDims.hpp>    // dim::Dim<N>

#include <alpaka/traits/mem/Buf.hpp>    // traits::Set
#include <alpaka/traits/Extents.hpp>    // traits::getXXX

#include <cassert>                      // assert

namespace alpaka
{
    //-----------------------------------------------------------------------------
    // Trait specializations for Set.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace mem
        {
            //#############################################################################
            //! The CUDA 1D memory set trait specialization.
            //#############################################################################
            template<>
            struct Set<
                alpaka::dim::Dim1,
                alpaka::mem::SpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TBuf>
                ALPAKA_FCT_HOST static auto set(
                    TBuf & buf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    
                    static_assert(
                        alpaka::mem::IsBufBase<TBuf>::value,
                        "The buffer has to be a base buffer!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBuf>, alpaka::dim::Dim1>::value,
                        "The destination buffer is required to have the dimensionality alpaka::dim::Dim1 for this specialization!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBuf>, alpaka::dim::DimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(buf));
                    assert(uiExtentWidth <= uiDstWidth);
                    
                    // Initiate the memory set.
                    ALPAKA_CUDA_CHECK(
                        cudaMemset(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(buf)),
                            static_cast<int>(byte),
                            uiExtentWidth * sizeof(alpaka::mem::ElemT<TBuf>)));
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf, 
                    typename TExtents>
                ALPAKA_FCT_HOST static auto set(
                    TBuf & buf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents,
                    cuda::detail::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    
                    static_assert(
                        alpaka::mem::IsBufBase<TBuf>::value,
                        "The buffer has to be a base buffer!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBuf>, alpaka::dim::Dim1>::value,
                        "The destination buffer is required to have the dimensionality alpaka::dim::Dim1 for this specialization!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBuf>, alpaka::dim::DimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(buf));
                    assert(uiExtentWidth <= uiDstWidth);
                    
                    // Initiate the memory set.
                    ALPAKA_CUDA_CHECK(
                        cudaMemsetAsync(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(buf)),
                            static_cast<int>(byte),
                            uiExtentWidth * sizeof(alpaka::mem::ElemT<TBuf>),
                            *stream.m_spCudaStream.get()));
                }
            };
            //#############################################################################
            //! The CUDA 2D memory set trait specialization.
            //#############################################################################
            template<>
            struct Set<
                alpaka::dim::Dim2,
                alpaka::mem::SpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TBuf>
                ALPAKA_FCT_HOST static auto set(
                    TBuf & buf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    
                    static_assert(
                        alpaka::mem::IsBufBase<TBuf>::value,
                        "The buffer has to be a base buffer!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBuf>, alpaka::dim::Dim2>::value,
                        "The destination buffer is required to have the dimensionality alpaka::dim::Dim2 for this specialization!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBuf>, alpaka::dim::DimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(buf));
                    auto const uiDstHeight(alpaka::extent::getHeight(buf));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);

                    // Initiate the memory set.
                    ALPAKA_CUDA_CHECK(
                        cudaMemset2D(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(buf)),
                            alpaka::mem::getPitchBytes(buf),
                            static_cast<int>(byte),
                            uiExtentWidth * sizeof(alpaka::mem::ElemT<TBuf>),
                            uiExtentHeight));
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TBuf>
                ALPAKA_FCT_HOST static auto set(
                    TBuf & buf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents,
                    cuda::detail::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    
                    static_assert(
                        alpaka::mem::IsBufBase<TBuf>::value,
                        "The buffer has to be a base buffer!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBuf>, alpaka::dim::Dim2>::value,
                        "The destination buffer is required to have the dimensionality alpaka::dim::Dim2 for this specialization!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBuf>, alpaka::dim::DimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(buf));
                    auto const uiDstHeight(alpaka::extent::getHeight(buf));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);

                    // Initiate the memory set.
                    ALPAKA_CUDA_CHECK(
                        cudaMemset2DAsync(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(buf)),
                            alpaka::mem::getPitchBytes(buf),
                            static_cast<int>(byte),
                            uiExtentWidth * sizeof(alpaka::mem::ElemT<TBuf>),
                            uiExtentHeight,
                            *stream.m_spCudaStream.get()));
                }
            };
            //#############################################################################
            //! The CUDA 3D memory set trait specialization.
            //#############################################################################
            template<>
            struct Set<
                alpaka::dim::Dim3,
                alpaka::mem::SpaceCuda>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TBuf>
                ALPAKA_FCT_HOST static auto set(
                    TBuf & buf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    
                    static_assert(
                        alpaka::mem::IsBufBase<TBuf>::value,
                        "The buffer has to be a base buffer!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBuf>, alpaka::dim::Dim3>::value,
                        "The destination base buffer is required to have the dimensionality alpaka::dim::Dim3 for this specialization!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBuf>, alpaka::dim::DimT<TExtents>>::value,
                        "The destination base buffer and the extents are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extents));
                    auto const uiExtentDepth(alpaka::extent::getDepth(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(buf));
                    auto const uiDstHeight(alpaka::extent::getHeight(buf));
                    auto const uiDstDepth(alpaka::extent::getDepth(buf));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);

                    // Fill CUDA parameter structures.
                    cudaPitchedPtr const cudaPitchedPtrVal(
                        make_cudaPitchedPtr(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(buf)),
                            alpaka::mem::getPitchBytes(buf),
                            uiDstWidth,
                            uiDstHeight));
                    
                    cudaExtent const cudaExtentVal(
                        make_cudaExtent(
                            uiExtentWidth,
                            uiExtentHeight,
                            uiExtentDepth));
                    
                    // Initiate the memory set.
                    ALPAKA_CUDA_CHECK(
                        cudaMemset3D(
                            cudaPitchedPtrVal,
                            static_cast<int>(byte),
                            cudaExtentVal));
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TBuf>
                ALPAKA_FCT_HOST static auto set(
                    TBuf & buf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents,
                    cuda::detail::StreamCuda const & stream)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    
                    static_assert(
                        alpaka::mem::IsBufBase<TBuf>::value,
                        "The buffer has to be a base buffer!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBuf>, alpaka::dim::Dim3>::value,
                        "The destination base buffer is required to have the dimensionality alpaka::dim::Dim3 for this specialization!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBuf>, alpaka::dim::DimT<TExtents>>::value,
                        "The destination base buffer and the extents are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extents));
                    auto const uiExtentDepth(alpaka::extent::getDepth(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(buf));
                    auto const uiDstHeight(alpaka::extent::getHeight(buf));
                    auto const uiDstDepth(alpaka::extent::getDepth(buf));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);

                    // Fill CUDA parameter structures.
                    cudaPitchedPtr const cudaPitchedPtrVal(
                        make_cudaPitchedPtr(
                            reinterpret_cast<void *>(alpaka::mem::getNativePtr(buf)),
                            alpaka::mem::getPitchBytes(buf),
                            uiDstWidth,
                            uiDstHeight));
                    
                    cudaExtent const cudaExtentVal(
                        make_cudaExtent(
                            uiExtentWidth,
                            uiExtentHeight,
                            uiExtentDepth));
                    
                    // Initiate the memory set.
                    ALPAKA_CUDA_CHECK(
                        cudaMemset3DAsync(
                            cudaPitchedPtrVal,
                            static_cast<int>(byte),
                            cudaExtentVal,
                            *stream.m_spCudaStream.get()));
                }
            };
        }
    }
}