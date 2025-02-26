import torch


def denosing(neuron_info: dict, target: str, source: str, top_k: int):
    """
    Args:
        neuron_info (dict): SAE neuron statistics information,
        target (str): eg. pos
        source (str): eg. neg

    Returns:
        dif_res (dic): 
        key: "latent_frequency" value frequency diffence
        key: "latent_value_mean" value mean value diffence
        key: "topk_freq_dif"  topk frequency difference neurons
    """
    assert target in neuron_info.keys() and source in neuron_info.keys(), (
        str(neuron_info.keys()) + "please check source and target"
    )
    # latent_frequency: neuron activation frequency
    # latent_value_mean:  neuron activation mean value
    
    dif_res = {
        "latent_frequency": torch.relu(
            neuron_info[target]["latent_frequency"]- neuron_info[source]["latent_frequency"]
        ),
        "latent_value_mean": torch.relu(
            neuron_info[target]["latent_value_mean"]- neuron_info[source]["latent_value_mean"]
        )
    }
    ########################### CORE IDEA
    """
    * 筛选那些在TARGET样本中被频繁激活，而在SOURCE样本中没有被频繁激活的神经元，我们将在下一步中手动激活这些目标神经元
    * Those neurons that are frequently activated in the TARGET samples but not frequently activated in the SOURCE samples. 
    """
    _, topk_freq_neuron_indices = torch.topk(dif_res["latent_frequency"], top_k)
    dif_res["topk_freq_dif"]=topk_freq_neuron_indices
    ########################### CORE IDEA
    assert not torch.all(
        dif_res["topk_freq_dif"] == 0
    ), "latent_frequency全为0元素读取有问题, check datapreprocess"
    return dif_res



