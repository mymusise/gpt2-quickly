from transformers import GPT2Tokenizer
import tensorflow as tf
from transformers import TFPerformerAttention
from transformers import GPT2Config
from transformers import TFGPT2MainLayer, TFGPT2LMHeadModel
from transformers.models.gpt2.modeling_tf_gpt2 import TFMLP, TFAttention, TFConv1D
from enum import Enum
from typing import Sequence, Optional, Union
from transformers.configuration_performer_attention import PerformerAttentionConfig, PerformerKernel, OrthogonalFeatureAlgorithm


class PerformerConfig(GPT2Config):

    def __init__(self,
                 attention_dropout: float = 0.1,
                 kernel_type: Union[str, PerformerKernel] = PerformerKernel.exp,
                 causal: bool = False,
                 use_recurrent_decoding: bool = False,
                 kernel_epsilon: float = 1e-4,
                 normalize_output: bool = True,
                 normalization_stabilizer: float = 1e-6,
                 use_linear_layers: bool = True,
                 linear_layer_names: Sequence[str] = ('q_linear', 'k_linear', 'v_linear', 'out_linear'),
                 num_random_features: Optional[int] = None,
                 use_thick_features: bool = False,
                 regularize_feature_norms: bool = True,
                 use_orthogonal_features: bool = True,
                 orthogonal_feature_algorithm: Union[str, OrthogonalFeatureAlgorithm] = OrthogonalFeatureAlgorithm.auto,
                 feature_redraw_interval: Optional[int] = 100,
                 redraw_stochastically: bool = False,
                 redraw_verbose: bool = False,
                 d_model: Optional[int] = None,
                 num_heads: Optional[int] = None,
                 **kwargs):

        self.attention_dropout = attention_dropout
        self.kernel_type = kernel_type
        self.causal = causal
        self.use_recurrent_decoding = use_recurrent_decoding
        self.kernel_epsilon = kernel_epsilon
        self.normalize_output = normalize_output
        self.normalization_stabilizer = normalization_stabilizer
        self.use_linear_layers = use_linear_layers
        self.linear_layer_names = linear_layer_names
        self.num_random_features = num_random_features
        self.use_thick_features = use_thick_features
        self.regularize_feature_norms = regularize_feature_norms
        self.use_orthogonal_features = use_orthogonal_features
        self.orthogonal_feature_algorithm = orthogonal_feature_algorithm
        self.feature_redraw_interval = feature_redraw_interval
        self.redraw_stochastically = redraw_stochastically
        self.redraw_verbose = redraw_verbose
        self.d_model = d_model
        self.num_heads = num_heads

        super().__init__(**kwargs)


class TFBlock(tf.keras.layers.Layer):
    def __init__(self, n_ctx, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * nx
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        self.c_attn = TFConv1D(config.n_embd * 3, nx, initializer_range=config.initializer_range, name="c_attn")
        self.attn = TFPerformerAttention(config, num_heads=config.n_head, d_model=config.n_embd, name="attn")
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_2")
        self.mlp = TFMLP(inner_dim, config, name="mlp")

    def call(self, x, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        a = self.ln_1(x)
        a = self.c_attn(a)
        query, key, value = tf.split(a, 3, axis=2)
        output_attn = self.attn(
            query, key, value)
        a = output_attn[0]  # output_attn: a, present, (attentions)
        x = x + a

        m = self.ln_2(x)
        m = self.mlp(m, training=training)
        x = x + m

        outputs = [x] + [x]
        return outputs  # x, present, (attentions)


class TFGPT2MainLayer(TFGPT2MainLayer):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.h = [TFBlock(config.n_ctx, config, scale=True, name="h_._{}".format(i)) for i in range(config.n_layer)]


class TFGPT2LMHeadModel(TFGPT2LMHeadModel):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        assert hasattr(config, 'attention_dropout')
        self.transformer = TFGPT2MainLayer(config, name="transformer")


# pconfig = PerformerConfig()
# configuration = GPT2Config()
# model = TFGPT2LMHeadModel(pconfig)

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# inputs = tokenizer("aa aa aa aa", return_tensors="tf")
# model(**inputs)
# out = model(**inputs)
# print(out[0].shape)
