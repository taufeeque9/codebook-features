from tqdm import tqdm
from codebook_features import models
from codebook_features import utils as cb_utils
import torch
import re

# We turn automatic differentiation off, to save GPU memory,
# as this tutorial focuses only on model inference
torch.set_grad_enabled(False)
torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_or_path = "EleutherAI/pythia-410m-deduped"
pre_trained_path = "taufeeque/pythia410m_wikitext_attn_codebook_model"

orig_cb_model = models.wrap_codebook(
    model_or_path=model_name_or_path, pretrained_path=pre_trained_path
)

orig_cb_model = orig_cb_model.to(DEVICE).eval()

hooked_kwargs = dict(
    center_unembed=False,
    fold_value_biases=False,
    center_writing_weights=False,
    fold_ln=False,
    refactor_factored_attn_matrices=False,
    device=DEVICE,
)

cb_model = models.convert_to_hooked_model(
    model_name_or_path, orig_cb_model, hooked_kwargs=hooked_kwargs
)
cb_model = cb_model.to(DEVICE).eval()
tokenizer = cb_model.tokenizer

sentence = "this is a random sentence to test."
input_tensor = tokenizer(sentence, return_tensors="pt")["input_ids"]
input_tensor = input_tensor.to(DEVICE)
output = orig_cb_model(input_tensor)["logits"]
hooked_output = cb_model(input_tensor)
assert torch.allclose(output, hooked_output, atol=1e-4)

very_first_head_cb = cb_model.blocks[0].attn.codebook_layer.codebook[0].codebook.weight

print(f"Codebook associated with the first head of the first layer of model: {very_first_head_cb}")

