#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>

// #ifndef AT_PER_OPERATOR_HEADERS
// #include <ATen/Functions.h>
// #include <ATen/NativeFunctions.h>
// #else
// Note: this file is generated
#include <ATen/ops/matmul.h>
// #endif

// #include <ATen/core/Tensor.h>


// asdfasdf

namespace at::native {
std::tuple<Tensor, Tensor> nat_attn(const Tensor &q, const Tensor & k, const Tensor & v){
    Tensor k_t = k.transpose(0, 1);
    Tensor x = q.matmul(k_t);
    auto a = x.tanh();
    auto o = at::matmul(a, v);
    // auto o = a.matmul(v);
    return std::make_tuple(o, a);
}

Tensor nat_attn_w_cpu(const Tensor &q, const Tensor & k, const Tensor & v){
    return zeros(3);
}


} // namespace at::native
