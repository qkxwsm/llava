Commands:
#CLI demo
python -m llava.serve.cli_one_image_per --model-path liuhaotian/llava-v1.5-7b --image-file test_images/cat.jpg test_images/dog.jpg test_images/giraffe.jpg 

#Menu dataset item
FOOD_IMAGES=/data/cb/dschaffe/vlm/llava/menu_images/MAFood121/images python3 test_menu_hf.py

#Training
cd duo_attention/
#echo HF_API_KEY=...
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true PYTHONPATH=/data/cb/dschaffe/vlm/llava/duo_attention:$PYTHONPATH/ ROOT_DIR=/data/cb/dschaffe/vlm/llava/ HF_HOME=~/.cache/huggingface/datasets  MASTER_ADDR=127.0.0.1 MASTER_PORT=12355 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 USE_LLAVA=1 python3 duo_attn/train.py --output_dir outputs_example --disable_wandb --dataset_format menu --context_lengths_num_intervals=50 --num_steps=50000 --context_length_min=5000 --context_length_max=5000 --num_passkeys=5 --max_length=10000 

#Benchmark
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true HF_HOME=~/.cache/huggingface/datasets PYTHONPATH=/data/cb/dschaffe/vlm/llava/duo_attention:$PYTHONPATH/ ROOT_DIR=/data/cb/dschaffe/vlm/llava/ MASTER_ADDR=127.0.0.1 MASTER_PORT=12355 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 USE_LLAVA=1 python3 duo_attn/eval/efficiency/benchmark_dynamic.py --disable_wandb --sparsity=0.75 --attn_load_dir outputs_50k/ --output_dir outputs_eval_example/
