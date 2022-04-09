NN_evaluation和NN_prediction會用到custom layer

from my_own_class.attention_with_context import AttentionWithContext
from my_own_class.layer_normalization import LayerNormalization
model = load_model(model_path, custom_objects={"AttentionWithContext": AttentionWithContext, "LayerNormalization":LayerNormalization})