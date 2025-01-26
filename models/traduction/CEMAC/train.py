from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 128,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "embed_tokens", "lm_head",],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = True,
    loftq_config = None,
)

import pandas as pd


df = pd.read_csv("fang.csv")

def format_translation(row):
  return f"""Traduire du français au fang:
Français: {row['Français']}
Fang: {row['Fang']}
{tokenizer.eos_token}"""


formatted_dataset = df.apply(format_translation, axis=1).tolist()


from datasets import Dataset
dataset = Dataset.from_dict({"text": formatted_dataset})



dataset = dataset.train_test_split(train_size = 0.01)["train"]

#dataset = dataset.map(formatting_prompts_func, batched = True,)

from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        max_steps = 120,
        warmup_steps = 10,

        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Mémoire maximale = {max_memory} Go.")
print(f"{start_gpu_memory} Go de mémoire réservée.")

trainer_stats = trainer.train()

from datasets import load_dataset

alpaca_dataset = load_dataset("FreedomIntelligence/alpaca-gpt4-korean", split="train")

print(alpaca_dataset[0])

alpaca_prompt = """다음은 작업을 설명하는 명령입니다. 요청을 적절하게 완료하는 응답을 작성하세요.

### 지침:
{}

### 응답:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(conversations):
    texts = []
    conversations = conversations["conversations"]
    for convo in conversations:
        text = alpaca_prompt.format(convo[0]["value"], convo[1]["value"]) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

alpaca_dataset = alpaca_dataset.map(formatting_prompts_func, batched = True,)

from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = alpaca_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        max_steps = 120,
        warmup_steps = 10,

        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} secondes utilisées pour l'entraînement.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes utilisées pour l'entraînement.")
print(f"Mémoire maximale réservée = {used_memory} Go.")
print(f"Mémoire maximale réservée pour l'entraînement = {used_memory_for_lora} Go.")
print(f"Pourcentage de la mémoire maximale réservée = {used_percentage} %.")
print(f"Pourcentage de la mémoire maximale réservée pour l'entraînement = {lora_percentage} %.")


#on passe   l'inference: recommandé après du (Instruction Finetuning).


FastLanguageModel.for_inference(model

inputs = tokenizer(
[ ("Traduis la phrase suivante en Fang:", 
        "Bonjour, comment allez-vous?",)
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
tokenizer.batch_decode(outputs)


model.save_pretrained_merged("lora_model", tokenizer, save_method="merged_16bit")
