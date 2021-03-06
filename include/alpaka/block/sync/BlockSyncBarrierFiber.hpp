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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED

#include <alpaka/block/sync/Traits.hpp>

#include <alpaka/core/Fibers.hpp>

#include <alpaka/core/Common.hpp>

#include <mutex>
#include <map>

namespace alpaka
{
    namespace block
    {
        namespace sync
        {
            //#############################################################################
            //! The thread id map barrier block synchronization.
            template<
                typename TIdx>
            class BlockSyncBarrierFiber
            {
            public:
                using BlockSyncBase = BlockSyncBarrierFiber;

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierFiber(
                    TIdx const & blockThreadCount) :
                        m_barrier(static_cast<std::size_t>(blockThreadCount)),
                        m_threadCount(blockThreadCount),
                        m_curThreadCount(static_cast<TIdx>(0u)),
                        m_generation(static_cast<TIdx>(0u))
                {}
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierFiber(BlockSyncBarrierFiber const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierFiber(BlockSyncBarrierFiber &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(BlockSyncBarrierFiber const &) -> BlockSyncBarrierFiber & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(BlockSyncBarrierFiber &&) -> BlockSyncBarrierFiber & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~BlockSyncBarrierFiber() = default;

                boost::fibers::barrier mutable m_barrier;

                TIdx mutable m_threadCount;
                TIdx mutable m_curThreadCount;
                TIdx mutable m_generation;
                int mutable m_result[2u];
            };

            namespace traits
            {
                //#############################################################################
                template<
                    typename TIdx>
                struct SyncBlockThreads<
                    BlockSyncBarrierFiber<TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto syncBlockThreads(
                        block::sync::BlockSyncBarrierFiber<TIdx> const & blockSync)
                    -> void
                    {
                        blockSync.m_barrier.wait();
                    }
                };

                //#############################################################################
                template<
                    typename TOp,
                    typename TIdx>
                struct SyncBlockThreadsPredicate<
                    TOp,
                    BlockSyncBarrierFiber<TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncBarrierFiber<TIdx> const & blockSync,
                        int predicate)
                    -> int
                    {
                        if(blockSync.m_curThreadCount == blockSync.m_threadCount)
                        {
                            blockSync.m_curThreadCount = static_cast<TIdx>(0u);
                            ++blockSync.m_generation;
                        }

                        auto const generationMod2(blockSync.m_generation % static_cast<TIdx>(2u));

                        // The first fiber will reset the value to the initial value.
                        if(blockSync.m_curThreadCount == static_cast<TIdx>(0u))
                        {
                            blockSync.m_result[generationMod2] = TOp::InitialValue;
                        }

                        ++blockSync.m_curThreadCount;

                        // We do not have to lock because there is only ever one fiber active per block.
                        blockSync.m_result[generationMod2] = TOp()(blockSync.m_result[generationMod2], predicate);

                        // After all block threads have combined their values ...
                        blockSync.m_barrier.wait();

                        // ... the result can be returned.
                        return blockSync.m_result[generationMod2];
                    }
                };
            }
        }
    }
}

#endif
