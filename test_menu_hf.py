from duo_attention.duo_attn.data_menu import MenuPriceRetrievalDataset
from transformers import AutoProcessor

import argparse

#Adapted from CLI interface
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

processor = AutoProcessor.from_pretrained(args.model_name)
image_processor, tokenizer = processor.image_processor, processor.tokenizer
#fake text dataset
words = ('hello', 'goodbye', 'word', 'thing', 'text', 'blank')
text_list = [(word + " ")*100 for word in words]
text_dataset = {"text":text_list} #get_dataset(...)

dataset = MenuPriceRetrievalDataset(
        text_dataset,
        #Get these from loading pretrained model
        tokenizer,
        image_processor,
        buffer_size=50,
        max_length=1000
        )
x = dataset.__getitem__(0)
print(x.keys())
#print(type(x['input_ids']))
print(tokenizer.decode(x['input_ids'].squeeze(), skip_special_tokens=True))

