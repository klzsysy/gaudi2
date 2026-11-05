"""Qwen3.5 prefill delta-rule CPU/HPU parity probe. Run without an HPU server."""
import torch
from comfy.text_encoders.qwen35 import torch_chunk_gated_delta_rule

torch.set_grad_enabled(False)
torch.manual_seed(42)
for length in [17, 65, 128]:
    q,k,v=[torch.randn(1,length,2,32,dtype=torch.bfloat16) for _ in range(3)]
    g=-torch.rand(1,length,2)
    beta=torch.rand(1,length,2)
    ref,state=torch_chunk_gated_delta_rule(q,k,v,g,beta,output_final_state=True)
    out,actual_state=torch_chunk_gated_delta_rule(*(x.to('hpu') for x in [q,k,v,g,beta]),output_final_state=True)
    out=out.cpu(); actual_state=actual_state.cpu()
    print(length,'finite',torch.isfinite(out).all().item(),'max_error',(out.float()-ref.float()).abs().max().item(),flush=True)
    torch.testing.assert_close(out,ref,rtol=0.03,atol=0.01)
    torch.testing.assert_close(actual_state,state,rtol=0.03,atol=0.01)
print('PASS')
