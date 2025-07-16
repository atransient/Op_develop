#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cutlass/cluster_launch.hpp"

#include <cute/tensor.hpp>

#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/util/debug.h"
#include "cutlass/device_kernel.h"

using namespace cute;

template <class ElementA,
          class SmemLayoutA,  // (M,K,P)
          uint32_t Stages>
struct SharedStorage
{
    array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
    typename cutlass::PipelineTmaAsync<Stages>::SharedStorage storage;
};
template <class ProblemShape, class CtaTiler, class TA, class SmemLayoutA, class TmaA, class TmaC, uint32_t NumStages>
__global__ void test0(ProblemShape gmem_shape, CtaTiler cta_tiler, TA const *A, CUTLASS_GRID_CONSTANT TmaA const tma_a, TA const *C, CUTLASS_GRID_CONSTANT TmaC const tma_c)
{
    //gmem tensor init
    Tensor mA = tma_a.get_tma_tensor(gmem_shape);
    Tensor mC = tma_c.get_tma_tensor(gmem_shape);
    Tensor gA = zipped_divide(mA, cta_tiler);
    Tensor gC = zipped_divide(mC, cta_tiler);

    // //smem init
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, SmemLayoutA, NumStages>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutA{});

    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                        group_modes<0,2>(sA), gA);
    
    auto [tCgC, tCsA] = tma_partition(tma_c, Int<0>{}, Layout<_1>{},
                                        group_modes<0,2>(sA), gC);

    //leader thread elect and warp id get
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_predicate = cute::elect_one_sync();

    //pipeline class define
    using MainloopPipeline = typename cutlass::PipelineTmaAsync<NumStages>;
    using PipelineState = typename cutlass::PipelineState<NumStages>;
    using BarrierType = typename MainloopPipeline::ProducerBarrierType;
    //pipeline init
    typename MainloopPipeline::Params params;
    uint32_t const TmaTransactionBytes = sizeof(TA) * size(cta_tiler);
    // if (thread0()) {
    //     print("mA  : "); print(mA); print("\n");
    //     print("gA  : "); print(gA); print("\n");
    //     print("sA  : "); print(sA); print("\n");
    //     print("tAgA  : "); print(tAgA); print("\n");
    //     print("tAsA  : "); print(tAsA); print("\n");
    //     print("TmaTransactionBytes  : "); print(TmaTransactionBytes); print("\n");
    //     print("---------------------\n");
    // }
    params.transaction_bytes = TmaTransactionBytes;
    params.is_leader = (lane_predicate && warp_idx == 0);
    params.num_consumers = blockDim.x;
    MainloopPipeline pipeline(smem.storage, params, Shape<_1, _1, _1>{});

    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
    __syncthreads();
    uint32_t num_iterations = size<1>(tAgA);
    uint32_t tile_count = num_iterations;
    uint32_t gA_read_count = 0;
    PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState smem_pipe_read;
    if (lane_predicate && (warp_idx == 0))
    {
        // print("smem_pipe_write:  ");print(smem_pipe_write);print("\n");
        int prologue_iterations = min(NumStages, num_iterations);
        for ( int i = 0; i < prologue_iterations; ++i)
        {
            pipeline.producer_acquire(smem_pipe_write);
            BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
            // Can also specify stage to commit directly
            copy(tma_a.with(*tma_barrier), tAgA(_,gA_read_count), tAsA(_,i));
            // pipeline.producer_commit(smem_pipe_write);
            
            --tile_count;
            ++smem_pipe_write;
            ++gA_read_count;
        }
    }
    if (thread0()) {
        print("tCsA(_, 0)  : "); print(tCsA(_, 0)); print("\n");
        print("tCgC(_, 0)  : "); print(tCgC(_, 0)); print("\n");
        print("---------------------\n");
    }
    uint write_c_index = 0;
    for (int i = 0; i < num_iterations; ++i)
    {
        pipeline.consumer_wait(smem_pipe_read);
        __syncthreads();
        uint read_index = smem_pipe_read.index();

        //printf smem data
        // cutlass::debug::dump_shmem(raw_pointer_cast(sA(_,_,read_index).data()), size(cta_tiler));
        copy(tma_c, tCsA(_, read_index), tCgC(_,write_c_index));
        ++write_c_index;
        pipeline.consumer_release(smem_pipe_read);

        if (tile_count != 0 && lane_predicate && warp_idx == 0)
        {
            // printf("111\n");
            uint write_index = smem_pipe_write.index();
            pipeline.producer_acquire(smem_pipe_write);
            BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
            // Can also specify stage to commit directly
            copy(tma_a.with(*tma_barrier), tAgA(_, gA_read_count), tAsA(_, write_index));
            // pipeline.producer_commit(smem_pipe_write);
            --tile_count;
            ++smem_pipe_write;
            ++gA_read_count;
        }
        ++smem_pipe_read;
    }
}


template <class TA>
void intermedia_config(int M, int N, TA const* A,TA const* C)
{
    auto gmem_shape = make_shape(M,N);
    auto bM = Int<64>{};
    auto bN = Int<32>{};
    auto num_stage = Int<2>{};
    auto dA = make_stride(Int<1>{}, M);
    auto cta_tiler = make_shape(bM, bN);
    
    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM, bN, num_stage));
    
    Tensor mA = make_tensor(A, make_shape(M,N), dA);
    Tensor mC = make_tensor(C, make_shape(M,N), dA);
    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bN));
    Copy_Atom tmaC = make_tma_atom(SM90_TMA_STORE{}, mC, sA(_,_,0), make_shape(bM,bN));
    int smem_size = int(sizeof(SharedStorage<TA, decltype(sA), num_stage>));
    dim3 block(128,1,1);
    dim3 dimCluster(1, 1, 1);
    dim3 grid(1,1,1);
    test0<decltype(gmem_shape), decltype(cta_tiler), TA, decltype(sA), decltype(tmaA), decltype(tmaC), num_stage><<<grid, block, smem_size>>>(gmem_shape, cta_tiler, A, tmaA, C, tmaC);
    CUTE_CHECK_LAST();
}

int main() {
    int m = 128;
    int n = 64;
    using TA = int;
    thrust::host_vector<TA> h_A(m*n);
    thrust::host_vector<TA> h_C(m*n);
    for (int j = 0; j < m*n; ++j) h_A[j] = TA(int(j%1000));
    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TA> d_C = h_C;
    intermedia_config<TA>(m,n,d_A.data().get(),d_C.data().get());
    CUTE_CHECK_LAST();
    thrust::host_vector<TA> cute_result = d_C;
    for (int i = 0; i < m*n; ++i)
    {
        if (h_A[i] != cute_result[i])
        {
            printf("index: %d, h_A val: %d, h_C val : %d\n", i, h_A[i], cute_result[i]);
            break;
        }
    }
    printf("finish\n");
}
