import torch
import transformers.modeling_utils
from transformers.generation.configuration_utils import GenerationConfig
from transformers.configuration_utils import PretrainedConfig

# 1. Parche para get_head_mask
def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    if head_mask is not None:
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask

if not hasattr(transformers.modeling_utils.PreTrainedModel, "get_head_mask"):
    transformers.modeling_utils.PreTrainedModel.get_head_mask = get_head_mask

# 2. Parche para torch.Tensor.to (Error de dtype=device)
orig_tensor_to = torch.Tensor.to
def patched_tensor_to(self, *args, **kwargs):
    if "dtype" in kwargs and kwargs["dtype"] is not None:
        if not isinstance(kwargs["dtype"], torch.dtype):
            kwargs["device"] = kwargs.pop("dtype")
    return orig_tensor_to(self, *args, **kwargs)
torch.Tensor.to = patched_tensor_to

# 3. PARCHE PARA RESTA DE BOOLEANOS
orig_sub = torch.Tensor.__sub__
def patched_sub(self, other):
    if self.dtype == torch.bool:
        return orig_sub(self.to(torch.int8), other)
    return orig_sub(self, other)
torch.Tensor.__sub__ = patched_sub

orig_rsub = torch.Tensor.__rsub__
def patched_rsub(self, other):
    if self.dtype == torch.bool:
        # Esto maneja el caso (1.0 - tensor_booleano)
        return orig_rsub(self.to(torch.int8), other)
    return orig_rsub(self, other)
torch.Tensor.__rsub__ = patched_rsub

# 4. Parche para torch.finfo
orig_finfo = torch.finfo
def patched_finfo(dtype=None):
    if dtype is not None and not isinstance(dtype, torch.dtype):
        return orig_finfo(torch.get_default_dtype())
    return orig_finfo(dtype)
torch.finfo = patched_finfo

# 5. Parche para Qwen Config
orig_from_model_config = GenerationConfig.from_model_config
@classmethod
def patched_from_model_config(cls, model_config, **kwargs):
    for attr in ["text_config", "vision_config", "decoder_config"]:
        if hasattr(model_config, attr):
            sub_config = getattr(model_config, attr)
            if isinstance(sub_config, dict):
                setattr(model_config, attr, PretrainedConfig.from_dict(sub_config))
    return orig_from_model_config(model_config, **kwargs)
GenerationConfig.from_model_config = patched_from_model_config