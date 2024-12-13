#from duo_attention.duo_attn.data_df import MultiplePasskeyRetrievalDataset, get_supervised_dataloader
from duo_attention.duo_attn.data_menu import MenuPriceRetrievalDataset
from duo_attention.duo_attn.data import get_supervised_dataloader
from transformers import AutoProcessor

import argparse

#Adapted from CLI interface
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch", type=int, default=4)
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

#dataset = MultiplePasskeyRetrievalDataset(text_dataset, tokenizer=tokenizer, buffer_size=50, max_length=1000, num_passkeys=2, passkey_length=2)
b = args.batch
dataloader = get_supervised_dataloader(dataset, processor, batch_size=b, num_workers=0, shuffle=False)
it = iter(dataloader)
results = next(it)
print("B")
#Print what we got
#dict_keys(['input_ids', 'labels', 'attention_mask', 'pixel_values'])
for key, tensor in results.items():
    print(key, "shape:", tensor.shape)

for i in range(b):
    print("\n\n\n\n")
    input_ids = results['input_ids'][i]
    print("# image tokens:", (input_ids == 32000).sum().item())
    print(input_ids[:50])
    print(tokenizer.decode(input_ids, skip_special_tokens=True))
    



