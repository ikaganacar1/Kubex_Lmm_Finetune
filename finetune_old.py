#!/usr/bin/env python3
"""
Fine-tuning script for Qwen3 model using Unsloth
Converted from Jupyter notebook to standalone Python script
"""

import os
import gc
import torch
import traceback
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

def setup_environment():
    """Set up environment variables for optimal training"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def print_gpu_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")

def formatting_prompts_func(examples):
    """Format ChatML conversations for training"""
    texts = []
    for conversation in examples["messages"]:
        text = ""
        for message in conversation:
            role = message["role"]
            content = message["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        texts.append(text)
    return {"text": texts}

def load_and_prepare_model(model_name, max_seq_length):
    """Load model with 4-bit quantization and add LoRA adapters"""
    print("Loading Qwen3:8B...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # 4-bit quantization
        trust_remote_code=True,
        use_cache=False,
        device_map="auto",
    )
    
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # Can use higher rank with 4-bit
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=8,
        lora_dropout=0,  # Unsloth recommends 0 for 4-bit
        bias="none",
        use_gradient_checkpointing="unsloth",  # Essential for 4-bit training
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model, tokenizer

def prepare_dataset(dataset_file):
    """Load and format the dataset"""
    dataset = load_dataset("json", data_files=dataset_file)["train"]
    print(f"Dataset size: {len(dataset)}")
    
    # Apply formatting to dataset
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # Verify the formatting worked
    print("Formatted sample:")
    print(dataset[0]["text"])
    
    return dataset

def create_training_config(batch_size, gradient_accumulation_steps, num_epochs, 
                          learning_rate, max_seq_length, output_dir):
    """Create training configuration"""
    return SFTConfig(
        # Basic settings
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        
        # Memory optimization - crucial for 4-bit
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",  # Use 8-bit optimizer with 4-bit model
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        output_dir=output_dir,
        report_to="none",
        
        # Other settings optimized for 4-bit
        warmup_steps=10,
        warmup_ratio=0.1,
        save_strategy="epoch",
        save_total_limit=1,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,  # Set to True can save memory but may affect quality
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        seed=42,
        
        # Additional 4-bit optimizations
        dataset_num_proc=1,  # Single process for stability
        per_device_eval_batch_size=batch_size,
    )

def test_model(model, tokenizer):
    """Test the trained model with a sample prompt"""
    print("\n🧪 Testing the model...")
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    # Test prompt
    messages = [
        {"role": "user", "content": "How do I create a Kubernetes service?"}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Generate
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Response:", response)

def main():
    """Main training function"""
    # ==================================================
    # SETTINGS 
    # ==================================================
    model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    max_seq_length = 512
    batch_size = 1
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    num_epochs = 1  # Start with 1 for testing
    output_dir = "./qwen3-finetuned"
    dataset_file = "kubernetes_chatml.jsonl"
    
    # Setup environment
    setup_environment()
    
    try:
        # Load and prepare model
        model, tokenizer = load_and_prepare_model(model_name, max_seq_length)
        
        # Prepare dataset
        dataset = prepare_dataset(dataset_file)
        
        # Clear memory before training
        torch.cuda.empty_cache()
        gc.collect()
        print_gpu_memory()
        
        # Create training configuration
        training_args = create_training_config(
            batch_size, gradient_accumulation_steps, num_epochs,
            learning_rate, max_seq_length, output_dir
        )
        
        # Clear memory again
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create trainer with Unsloth's optimizations
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            dataset_num_proc=1,
        )
        
        # Check memory usage
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            print(f"GPU: {gpu_stats.name}")
            print(f"Total Memory: {gpu_stats.total_memory / 1024**3:.1f} GB")
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"Allocated before training: {allocated:.2f} GB")
            print(f"Free memory: {(gpu_stats.total_memory / 1024**3) - allocated:.2f} GB")
        
        # Train with 4-bit optimization
        print("\nTraining started...")
        trainer_stats = trainer.train()
        
        print("\n✅ 4-bit training complete!")
        if hasattr(trainer_stats, 'training_loss'):
            print(f"Final loss: {trainer_stats.training_loss:.4f}")
        
        # ==================================================
        # SAVE 4-BIT TRAINED MODEL
        # ==================================================
        print("\nSaving 4-bit trained model...")
        
        # Save in 16-bit for better compatibility
        # Unsloth can merge and save in 16-bit even from 4-bit training
        model.save_pretrained_merged(
            output_dir, 
            tokenizer,
            save_method="merged_16bit",  # Options: "merged_16bit", "merged_4bit", "lora"
        )
        print(f"✅ Model saved to {output_dir}")
        
        # Test the model
        test_model(model, tokenizer)
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ Out of memory error during 4-bit training!")
        print("\nMemory-saving solutions for 4-bit training:")
        print("1. Use smaller model: 'unsloth/Qwen2.5-3B-bnb-4bit' or 'unsloth/tinyllama-bnb-4bit'")
        print("2. Reduce max_seq_length to 1024 or 512")
        print("3. Enable packing: packing=True in training_args")
        print("4. Increase gradient_accumulation_steps to 8 or 16")
        print("5. Reduce LoRA rank r to 8 or 4")
        print(f"\nCurrent memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
        raise

    except TypeError as e:
        if "precision" in str(e).lower():
            print(f"\n❌ Precision mismatch error: {e}")
            print("\nFix: Add these lines to training_args:")
            print("bf16=True, fp16=False  # If GPU supports bf16")
            print("OR")
            print("bf16=False, fp16=True  # If GPU doesn't support bf16")
        raise

    except Exception as e:
        print(f"\n❌ Training error: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()