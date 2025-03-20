#ifndef EMBEDDING_H
#define EMBEDDING_H
#include "layer.h"
#include <base/cuda_config.h>

struct EmbeddingOutput {
  tensor::Tensor input_tokens;
  tensor::Tensor input_embeddings;
  tensor::Tensor input_token_num;
};

class EmbeddingLayer : public LayerFp32Param {
 public:
  explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                          int32_t vocab_size);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t vocab_size_ = 0;
};

#endif //EMBEDDING_H
