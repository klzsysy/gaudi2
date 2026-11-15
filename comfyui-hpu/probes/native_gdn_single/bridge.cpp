#include <torch/library.h>
#include <ATen/ATen.h>
#include <hpu_custom_op_pt2.h>

habana::PartialOutputMetaDataVector metadata(const at::Stack& stack) {
    const auto state = stack.at(0).toTensor();
    const auto query = stack.at(5).toTensor();
    return {{at::kFloat, state.sizes().vec()}, {at::kFloat, {state.size(0), query.size(2), state.size(1), state.size(2)}}};
}

std::tuple<at::Tensor, at::Tensor> execute(const at::Tensor& state, const at::Tensor& pk, const at::Tensor& pv,
    const at::Tensor& pd, const at::Tensor& pb, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& d, const at::Tensor& b, const at::Tensor& program) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::qwen_gdn_single");
    auto result = op.execute({state, pk, pv, pd, pb, q, k, v, d, b, program});
    return {result[0], result[1]};
}

TORCH_LIBRARY_FRAGMENT(custom_op, library) {
    library.def("qwen_gdn_single(Tensor state, Tensor pk, Tensor pv, Tensor pd, Tensor pb, Tensor q, Tensor k, Tensor v, Tensor d, Tensor b, Tensor program) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::qwen_gdn_single", "qwen_gdn_single_f32", metadata, nullptr);
}
TORCH_LIBRARY_IMPL(custom_op, HPU, library) { library.impl("qwen_gdn_single", execute); }
