from transformer_lens import HookedTransformer
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from dotenv import load_dotenv
load_dotenv("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env")

# Choose a layer you want to focus on
# For this tutorial, we're going to use layer 2
device="cpu"
layer = 6

# get model
model = HookedTransformer.from_pretrained("gemma-2b", device=device)

# get the SAE for this layer
sae, cfg_dict, _ = SAE.from_pretrained(
    release="gemma-2b-res-jb", sae_id=f"blocks.{layer}.hook_resid_post", device=device
)

# get hook point
hook_point = sae.cfg.hook_name
print(hook_point)