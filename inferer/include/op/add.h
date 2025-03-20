#ifndef ADD_H
#define ADD_H
#include <op/layer.h>
#include <base/base.h>
#include <base/cuda_config.h>

namespace op {
    class VecAddLayer : public Layer {
    public:
        explicit VecAddLayer(base::DeviceType device_type);

        base::Status check() const override;

        base::Status forward() override;

    };

}

#endif //ADD_H
