"""
Lab 4.1: Prompting + LoRA Fine-Tuning (FIXED VERSION)
====================================================
Objective: Prompt engineering + LoRA fine-tuning pada small LLM (1-3B params)
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ============================================================
# 0. PLATFORM DETECTION
# ============================================================
def get_device():
    if torch.cuda.is_available():
        print("✓ CUDA GPU terdeteksi — mode Colab T4")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("✓ Apple MPS terdeteksi — mode MacBook M-series")
        return "mps"
    else:
        print("⚠ Tidak ada GPU — mode CPU (lambat, gunakan model kecil)")
        return "cpu"

DEVICE = get_device()

# ============================================================
# 1. CONFIG
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 
OUTPUT_DIR = "./lora_finetuned_model"

# ============================================================
# PART B: LoRA FINE-TUNING
# ============================================================
def part_b_lora_finetuning():
    print("\n" + "="*60)
    print("PART B: LoRA FINE-TUNING")
    print("="*60)

    # [1/5] Load Model & Tokenizer
    print(f"\n[1/5] Loading model: {MODEL_NAME}")
    
    torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map=DEVICE if DEVICE != "cpu" else None
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # [2/5] Apply LoRA Adapters
    print("\n[2/5] Applying LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                  
        lora_alpha=32,         
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] 
    )
    
    # Catatan: JANGAN panggil get_peft_model() di sini.
    # Versi trl baru: SFTTrainer sendiri yang akan membungkus model
    # dengan LoRA saat kita kasih `peft_config=` ke SFTTrainer di bawah.
    # Kalau di-wrap manual di sini, nanti error "double wrap".

    # [3/5] Prepare Toy Dataset
    print("\n[3/5] Preparing dataset...")
    toy_data = [
        {"text": "<|user|>\nApa itu diffusion model?<|end|>\n<|assistant|>\nDiffusion model adalah model generatif yang belajar membalikkan proses penambahan noise secara sekuensial untuk menciptakan data baru.<|end|>"},
        {"text": "<|user|>\nBagaimana cara kerja LoRA?<|end|>\n<|assistant|>\nLoRA membekukan bobot asli model dan menyisipkan matriks dekomposisi peringkat rendah yang dapat dilatih ke dalam setiap lapisan Transformer.<|end|>"},
    ]
    
    dataset = Dataset.from_list(toy_data)

# [4/5] Configure SFTTrainer
    print("\n[4/5] Configuring SFTTrainer...")

    # Versi trl baru (>=0.12) WAJIB pakai SFTConfig, bukan TrainingArguments biasa.
    # dataset_text_field & max_seq_length juga pindah ke sini.
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        # trl baru sudah tidak menerima `tokenizer=`, gantinya `processing_class=`
        processing_class=tokenizer,
    )

    # Sekarang trainer.model sudah PeftModel (LoRA sudah ter-apply oleh SFTTrainer)
    trainer.model.print_trainable_parameters()
    # [5/5] Train & Save
    print("\n[5/5] Starting fine-tuning...")
    trainer.train()
    
    print(f"\n✓ Training selesai! Menyimpan adapter ke: {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    return trainer.model, tokenizer

if __name__ == "__main__":
    part_b_lora_finetuning()
    print("\n✓ Lab 4.1 selesai dengan sukses!")