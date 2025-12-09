import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig # SFTConfig is not strictly needed for this fix but kept for completeness
import sys
from dotenv import load_dotenv

# Load variables from .env file into os.environ
load_dotenv()

# --- Configuration ---
# 1. MODEL AND DATA PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_FILE_PATH = os.path.join(BASE_DIR, "llama_training_data.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "llama_3_2_3b_finetuned")

# The Hugging Face ID for the Llama 3.2 3B model
MODEL_NAME = "meta-llama/Llama-3.2-3B" 

# 2. QLoRA Parameters
LORA_R = 16          # Rank of the update matrices.
LORA_ALPHA = 32      # Scaling factor for the weights.
LORA_DROPOUT = 0.05  # Dropout probability.

# 3. Training Arguments (Adjust based on your GPU)
NUM_EPOCHS = 3.0
PER_DEVICE_BATCH_SIZE = 2 # Start low (1 or 2)
GRADIENT_ACCUMULATION_STEPS = 4 # Simulates a batch size of 8 (2 * 4)
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512 # Max length of the conversation string

# ----------------- Main Fine-Tuning Function -----------------

if os.getenv("HF_TOKEN") is None:
    print("❌ FATAL: HF_TOKEN not found in environment or .env file.")
    print("Please ensure your .env file is correct and try again.")
    sys.exit(1)

def fine_tune_llama():
    """Load data and model, then run the QLoRA fine-tuning process."""
    
    print(f"--- 🚀 Starting Llama 3.2 3B Fine-Tuning ---")
    
    # Check if data file exists
    if not os.path.exists(TRAINING_FILE_PATH):
        print(f"❌ FATAL: Training data file not found at {TRAINING_FILE_PATH}")
        sys.exit(1)
        
    # 1. Load Data
    try:
        # Load the JSONL file. 'text' is the column name containing the instruction format.
        dataset = load_dataset("json", data_files=TRAINING_FILE_PATH, split="train")
        print(f"✅ Dataset loaded successfully. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)


    # 2. Load Model and Tokenizer with 4-bit Quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    print(f"Loading Model: {MODEL_NAME} with 4-bit Quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0} # Use the first available GPU
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right" # Important for CausalLM (decoding)

    # 3. Define LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
    )

    # 4. Define Training Arguments
    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim="paged_adamw_8bit",
        save_steps=500,
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        fp16=False,
        bf16=True, 
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        disable_tqdm=False, 
    )

    # 5. Initialize SFT Trainer
    # --- FIX APPLIED HERE: Removed the problematic keyword arguments ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        # tokenizer=tokenizer,
        args=training_arguments,
        # NOTE: Using the internal ConstantLengthDataset requires the following two arguments:
        # If the fix above fails, uncomment these two lines, and if they still fail, 
        # you MUST downgrade your TRL version (e.g., to 0.7.10 or lower).
        # dataset_text_field="text", 
        # max_seq_length=MAX_SEQ_LENGTH,
    )
    
    # 6. Start Training
    print("\n--- Starting Fine-Tuning ---")
    trainer.train()

    # 7. Save the final adapter weights
    print(f"\n--- Saving final adapter weights to {OUTPUT_DIR} ---")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n✅ Llama 3.2 3B Fine-Tuning Complete!")


if __name__ == "__main__":
    if torch.cuda.is_available():
        fine_tune_llama()
    else:
        print("❌ CUDA not found. Please ensure PyTorch and a compatible GPU are installed.")