#include <torch/library.h>
#include <ATen/ATen.h>
#include <hpu_custom_op_pt2.h>

using habana::PartialOutputMetaDataVector;

PartialOutputMetaDataVector output_meta(const at::Stack& stack) {
    auto input = stack.at(0).toTensor();
    auto shape = input.sizes().vec();
    shape.back() /= 2;
    return {{at::kBFloat16, shape}};
}

PartialOutputMetaDataVector quant_meta(const at::Stack& stack) {
    auto input = stack.at(0).toTensor();
    return {{at::kFloat8_e4m3fn, {input.size(0), input.size(1) / 2}}, {at::kFloat, {input.size(0), 1}}};
}

at::Tensor situ(const at::Tensor& input) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::moss_situ");
    return op.execute({input})[0];
}

std::tuple<at::Tensor, at::Tensor> situ_quant(const at::Tensor& input) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::moss_situ_quant");
    auto output = op.execute({input});
    return {output[0], output[1]};
}

TORCH_LIBRARY_FRAGMENT(custom_op, library) {
    library.def("moss_situ(Tensor input) -> Tensor");
    library.def("moss_situ_quant(Tensor input) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::moss_situ", "moss_situ_fast_bf16", output_meta, nullptr);
    habana::custom_op::registerUserCustomOp("custom_op::moss_situ_quant", "moss_situ_quant_fast_bf16", quant_meta, nullptr);
}

TORCH_LIBRARY_IMPL(custom_op, HPU, library) {
    library.impl("moss_situ", situ);
    library.impl("moss_situ_quant", situ_quant);
}
