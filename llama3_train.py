#%%
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import argparse
from datasets import Dataset

def formatting_prompts_func(examples):
    alpaca_prompt = """
    ### 질문:
    {}
    ### 답변:
    {}"""

    EOS_TOKEN = '<|end_of_text|>' # Must add EOS_TOKEN
    inputs       = examples["question"]
    outputs      = examples["answer"]
    texts = []
    
    for input, output in zip( inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    
    return { "text" : texts, }

parser = argparse.ArgumentParser(description='LLAMA3 QLora Training')
parser.add_argument('--model_name', type = str, default="unsloth/llama-3-8b-bnb-4bit")
parser.add_argument('--max_seq_length', type = int, default=2048)
parser.add_argument('--r', type = int, default=16)
parser.add_argument('--lora_alpha', type = int, default=16)
parser.add_argument('--lora_dropout', type = float, default=0)
parser.add_argument('--random_state', type = int, default=3407)
parser.add_argument('--dataset_path', type = str, default="")
parser.add_argument('--per_device_train_batch_size', type = int, default=1)
parser.add_argument('--gradient_accumulation_steps', type = int, default=1)
parser.add_argument('--warmup_steps', type = int, default=5)
parser.add_argument('--max_steps', type = int, default=19320)
parser.add_argument('--learning_rate', type = float, default=0.0002)
parser.add_argument('--logging_steps', type = int, default=1)
parser.add_argument('--save_steps', type = int, default=500)
parser.add_argument('--save_total_limit', type = int, default=3)
parser.add_argument('--output_dir', type = str, default="./outputs")


if __name__ == '__main__':
    args = parser.parse_args()
    model_name = args.model_name
    max_seq_length = args.max_seq_length
    r = args.r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    random_state = args.random_state
    dataset_path = args.dataset_path
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    warmup_steps = args.warmup_steps
    max_steps = args.max_steps
    learning_rate = args.learning_rate
    logging_steps = args.logging_steps
    save_steps = args.save_steps
    save_total_limit = args.save_total_limit
    output_dir = args.output_dir

    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = load_in_4bit,
        device_map = 'auto',)
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = random_state,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        )

    dataset = Dataset.from_json(dataset_path)
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = per_device_train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_steps = warmup_steps,
            max_steps = max_steps,
            learning_rate = learning_rate,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = logging_steps,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
            ),
        )
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    trainer_stats = trainer.train()
    trainer.model.save_pretrained(output_dir)
# %%
