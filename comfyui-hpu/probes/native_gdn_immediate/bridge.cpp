#include <torch/library.h>
#include <ATen/ATen.h>
#include <hpu_custom_op_pt2.h>

habana::PartialOutputMetaDataVector metadata(const at::Stack& stack) {
    const auto state = stack.at(0).toTensor();
    const auto value = stack.at(3).toTensor();
    return {{at::kFloat, state.sizes().vec()}, {at::kFloat, value.sizes().vec()}};
}

std::tuple<at::Tensor, at::Tensor> execute(const at::Tensor& state, const at::Tensor& q,
    const at::Tensor& k, const at::Tensor& v, const at::Tensor& d, const at::Tensor& b) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::qwen_gdn_immediate");
    auto result = op.execute({state, q, k, v, d, b});
    return {result[0], result[1]};
}

TORCH_LIBRARY_FRAGMENT(custom_op, library) {
    library.def("qwen_gdn_immediate(Tensor state, Tensor q, Tensor k, Tensor v, Tensor d, Tensor b) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::qwen_gdn_immediate", "qwen_gdn_immediate_f32", metadata, nullptr);
}
TORCH_LIBRARY_IMPL(custom_op, HPU, library) { library.impl("qwen_gdn_immediate", execute); }
