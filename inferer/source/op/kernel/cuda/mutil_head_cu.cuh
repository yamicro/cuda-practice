//
// Created by yami on 25-3-29.
//

#ifndef MUTIL_HEAD_CU_CUH
#define MUTIL_HEAD_CU_CUH
namespace kernel {
void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config);
}

#endif //MUTIL_HEAD_CU_CUH
