//
// Created by yami on 25-3-16.
//

#include "kernel_inter.h"
#include "cpu/add_kernel.h"
#include "cpu/rmsnorm.h"
#include "cpu/embedding.h"
#include "cpu/matul.h"
#include "cpu/rope.h"
#include "cpu/scaler.h"
#include "cpu/scaler_sum.h"
#include "cpu/softmax.h"
#include "cpu/swiglu.h"
#include "cuda/add_kernel_cu.cuh"
#include "cuda/rmsnorm_cu.cuh"
#include "cuda/matul.cuh"
#include "cuda/embedding_cu.cuh"
#include "cuda/rope_cu.cuh"
#include "cuda/swiglu_cu.cuh"
#include "cuda/softmax_cu.cuh"



namespace kernel {
    AddKernel get_add_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return add_kernel_cpu;
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            return add_kernel_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get a add kernel.";
            return nullptr;
        }
    }

    RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type)
    {
        if (device_type == base::DeviceType::kDeviceCPU)
        {
            return rmsnorm_kernel_cpu;
        } else if (device_type == base::DeviceType::kDeviceCUDA)
        {
            return rmsnorm_kernel_cu;
        }else
        {
            LOG(FATAL) << "Unknown device type for get a rms norm kernel.";
            return nullptr;
        }
    }

    MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return matmul_kernel_cpu;
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            return matmul_kernel_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get an matmul kernel.";
            return nullptr;
        }
    }

    EmbeddingKernel get_emb_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return emb_kernel_normal;
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            return emb_kernel_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get an embedding kernel.";
            return nullptr;
        }
    }

    RoPEKernel get_rope_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return rope_kernel_cpu;
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            return rope_kernel_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get a rope kernel.";
            return nullptr;
        }
    }

    ScaleKernel get_scale_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return scale_inplace_cpu;
        } else {
            LOG(FATAL) << "Unknown device type for get a rope kernel.";
            return nullptr;
        }
    }

    SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return softmax_inplace_cpu;
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            return softmax_inplace_cu;
        }
    }

    SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void* stream) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return swiglu_kernel_cpu;
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            return swiglu_kernel_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
            return nullptr;
        }
    }

    ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return scale_sum_kernel_cpu;
        } else {
            LOG(FATAL) << "Unknown device type for get a scale and reduce kernel.";
            return nullptr;
        }
    }
}
