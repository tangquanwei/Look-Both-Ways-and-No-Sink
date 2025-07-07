from .modeling_llama import LlamaModel
from .modeling_mistral import MistralModel
from .modeling_qwen2 import Qwen2Model
from .modeling_olmo import OlmoModel, OlmoForTokenClassification
from .modeling_phi3 import Phi3Model
from .modeling_utils import *

from transformers import AutoModelForTokenClassification, OlmoConfig
import transformers.models

from .modeling_flash_attention_utils import _flash_attention_forward
transformers.models.llama.modeling_llama._flash_attention_forward = _flash_attention_forward

AutoModelForTokenClassification.register(OlmoConfig, OlmoForTokenClassification)

transformers.models.llama.modeling_llama.LlamaModel = LlamaModel
transformers.models.mistral.modeling_mistral.MistralModel = MistralModel
transformers.models.qwen2.modeling_qwen2.Qwen2Model = Qwen2Model
transformers.models.phi3.modeling_phi3.Phi3Model = Phi3Model
transformers.models.olmo.modeling_olmo.OlmoModel = OlmoModel