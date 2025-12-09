import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

# Load HF_TOKEN from .env for base model access
load_dotenv()

# --- Configuration (MUST match finetune_llama.py) ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B" 
LORA_ADAPTER_DIR = "llama_3_2_3b_finetuned" # Output folder from finetuning script
MERGED_MODEL_DIR = "llama_3_2_3b_presales_merged" # Final directory for Ollama conversion

def merge_and_save_model():
    """Loads the base model, loads LoRA weights, merges them, and saves the final full model."""

    if not os.path.exists(LORA_ADAPTER_DIR):
        print(f"❌ FATAL: Adapter weights not found at {LORA_ADAPTER_DIR}. Run finetune_llama.py first.")
        sys.exit(1)
        
    print(f"--- 🚀 Starting Merge Process ---")
    
    # 1. Load the Base Model (using HF_TOKEN from environment for gated access)
    print(f"Loading base model: {BASE_MODEL_ID} in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16, # Use higher precision for merging
        device_map="auto",
        # Hugging Face will automatically use the HF_TOKEN environment variable
    )
    
    # 2. Load the PEFT (LoRA) Adapter Weights
    print(f"Loading LoRA adapter from: {LORA_ADAPTER_DIR}")
    peft_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    
    # 3. Merge the LoRA weights into the base model
    print("Merging adapter weights into the base model (this takes time)...")
    merged_model = peft_model.merge_and_unload()
    
    # 4. Load and save the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # 5. Save the final merged model and tokenizer
    os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
    print(f"Saving merged model and tokenizer to: {MERGED_MODEL_DIR}")
    merged_model.save_pretrained(MERGED_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)
    
    print("\n✅ Merge Complete! Directory created:", MERGED_MODEL_DIR)

if __name__ == "__main__":
    if torch.cuda.is_available():
        merge_and_save_model()
    else:
        print("❌ CUDA not found. Cannot run merge process.")
