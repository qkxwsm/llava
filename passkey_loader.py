import os
import torch
import json
import gzip
from pathlib import Path
from datasets import Dataset
from torchvision import transforms
from typing import Dict, Callable
from PIL import Image
from transformers import AutoProcessor
from typing import List, Dict, Any
from duo_attention.duo_attn.data import get_dataset

class PasskeyDataset(torch.utils.data.Dataset):
    def __init__(self, processor: AutoProcessor):
        super(PasskeyDataset, self).__init__()
        
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
                
        self.all_images = [self.transform(Image.open(os.path.join(os.path.join(os.environ["ROOT_DIR"], "augmented_images"), file))) 
                        for file in os.listdir(os.path.join(os.environ["ROOT_DIR"], "augmented_images"))]
        self.all_prompts = ["This is a placeholder prompt. Answer how you see fit." for _ in range(len(self.all_images))]
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        tokenized_input = self.processor(images=self.all_images[idx], text=self.all_prompts[idx], padding=True, do_rescale=False, return_tensors="pt")
        processor_tuple = (tokenized_input["input_ids"].tolist(), tokenized_input["attention_mask"].tolist(), tokenized_input["pixel_values"].tolist())
        return processor_tuple, self.all_prompts[idx] # by default, just make the VLM mimic its text output


def save_as_hf_dataset_advanced(
    dataset: torch.utils.data.Dataset,
    output_path: str,
    special_handlers: Dict[str, Callable] = None
) -> None:
    """
    Advanced version with custom type handling.
    
    Args:
        dataset: PyTorch Dataset object
        output_path: Path for output JSON file
        feature_names: List of feature names
        special_handlers: Dict mapping feature names to conversion functions
    """
    special_handlers = special_handlers or {}
    data_dicts = []
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        
        if isinstance(item, tuple):
            item_dict = {"tokenized_input" : item[0], "response": item[1]}
            print(f"Iteration {idx}: {type(item[0][0])}, {type(item[0][1])}, {type(item[0][2])}, {type(item[1])}")
        elif isinstance(item, dict):
            item_dict = {}
            for k, v in item.items():
                if k in special_handlers:
                    v = special_handlers[k](v)
                elif isinstance(v, torch.Tensor):
                    v = v.tolist()
                item_dict[k] = v
                
        data_dicts.append(item_dict)

    
    with gzip.open(output_path, 'wt') as f:
        for element in data_dicts:
            json.dump({**element, 'split': 'train'}, f)
    
    #save_dataset_compressed(data_dicts, output_path)   


def save_dataset_compressed(data_dicts: List[Dict[str, Any]], output_path: str):
    dataset = Dataset.from_list(data_dicts)
    dataset.save_to_disk(output_path, compression='zstd')



if __name__ == '__main__':
    if 'DECOMPRESS' not in os.environ.keys():
        proc = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
        output_dir = os.path.join(os.environ['ROOT_DIR'], 'passkey_images.json.gz')
        save_as_hf_dataset_advanced(PasskeyDataset(proc), output_dir)
    else:
        get_dataset(os.path.join(os.environ["ROOT_DIR"], 'passkey_images.json.gz'))