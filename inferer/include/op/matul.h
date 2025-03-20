#ifndef MATUL_H
#define MATUL_H
#include "layer.h"
#include <base/cuda_config.h>
namespace op {
    class MatmulLayer : public LayerFp32Param {
    public:
        explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1);

        base::Status check() const override;

        base::Status forward() override;

    private:
        int32_t dim0_ = 0;
        int32_t dim1_ = 0;
    };
}
#endif //MATUL_H
