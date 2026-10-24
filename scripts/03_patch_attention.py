"""补丁 2：T4(sm_75) 不支持 FlashAttention / cuDNN attention，强制回退 SDPA。

上游 cuda_graph.py 只检查算子是否存在、不检查算力等级，导致 T4 上会调用
torch.ops.aten._flash_attention_forward 并报
`FlashAttention only supports Ampere GPUs or newer`。
"""
import importlib.util
from pathlib import Path

OLD_FLASH = (
    '        flash = fused and config.head_dim <= 256 and hasattr(torch.ops.aten, "_flash_attention_forward") and (\n'
    '            "seqused_k" in str(torch.ops.aten._flash_attention_forward.default._schema))'
)
NEW_FLASH = (
    '        flash = (fused and torch.cuda.get_device_capability(self.device)[0] >= 8\n'
    '                 and config.head_dim <= 256 and hasattr(torch.ops.aten, "_flash_attention_forward") and\n'
    '                 "seqused_k" in str(torch.ops.aten._flash_attention_forward.default._schema))'
)
OLD_SELECT = (
    '            attention_backend = "flash" if flash else "cudnn" if fused and torch.backends.cudnn.is_available() else "sdpa"'
)
NEW_SELECT = '            attention_backend = "flash" if flash else "sdpa"'

roots = []
spec = importlib.util.find_spec("yue2")
if spec and spec.submodule_search_locations:
    roots.extend(Path(p) for p in spec.submodule_search_locations)
local = Path("/content/YuE/src/yue2")
if local.is_dir():
    roots.append(local)

for root in dict.fromkeys(roots):
    path = root / "cuda_graph.py"
    text = path.read_text(encoding="utf-8")
    new = text.replace(OLD_FLASH, NEW_FLASH).replace(OLD_SELECT, NEW_SELECT)
    if new != text:
        path.write_text(new, encoding="utf-8")
    ok = NEW_FLASH in new and NEW_SELECT in new
    print(root, "| attention patch applied:", ok)
    if not ok:
        raise SystemExit("补丁未生效，上游代码可能已变更")
