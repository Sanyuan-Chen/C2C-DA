__version__ = "2.0.0"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')
    absl.logging._warn_preinit_stderr = False
except:
    pass

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Files and general utilities
from .file_utils import (TRANSFORMERS_CACHE, PYTORCH_TRANSFORMERS_CACHE, PYTORCH_PRETRAINED_BERT_CACHE,
                         cached_path, add_start_docstrings, add_end_docstrings,
                         WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, CONFIG_NAME,
                         is_tf_available, is_torch_available)

# Tokenizers
from .tokenization_utils import (PreTrainedTokenizer)
from .tokenization_gpt2 import GPT2Tokenizer

# Configurations
from .configuration_utils import PretrainedConfig
from .configuration_gpt2 import GPT2Config, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP

# Modeling
if is_torch_available():
    from .modeling_utils import (PreTrainedModel, prune_layer, Conv1D)
    from .modeling_gpt2 import (GPT2PreTrainedModel, GPT2Model,
                                GPT2LMHeadModel, GPT2DoubleHeadsModel, GPT2SurfaceRealizationHeadsModel,
                                load_tf_weights_in_gpt2, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP)

    # Optimization
    from .optimization import (AdamW, ConstantLRSchedule, WarmupConstantSchedule, WarmupCosineSchedule,
                               WarmupCosineWithHardRestartsSchedule, WarmupLinearSchedule)
