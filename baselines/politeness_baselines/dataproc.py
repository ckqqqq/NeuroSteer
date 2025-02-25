from datasets import load_dataset
path = "/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/style_transfer/politeness-corpus"

def load_and_prepare_polite_dataset(dataset_path: str, seed: int):
    # logging.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    dataset["train"] = dataset['train'].shuffle(seed=seed)
    
    total_samples = len(dataset['train'])
    val_set = dataset['train'].select(range(total_samples//2, total_samples - total_samples//4))
    test_set = dataset['train'].select(range(total_samples - total_samples//4, total_samples))
    train_set = dataset['train'].select(range(total_samples//2))
    print(train_set,val_set,test_set)
    
    # logging.info("Filtering dataset for polite and nonpolite samples")
    pos_train_set = train_set.filter(lambda example: example['label'] == 2)
    neg_train_set = train_set.filter(lambda example: example['label'] == 0)
    
    # logging.info(f"Selected {len(pos_train_set)} polite and {len(neg_train_set)} nonpolite samples")
    assert len(val_set) != 0 and len(test_set) != 0 and len(pos_train_set) != 0 and len(neg_train_set) != 0, "数据集不兼容"
    return pos_train_set, neg_train_set, val_set, test_set
