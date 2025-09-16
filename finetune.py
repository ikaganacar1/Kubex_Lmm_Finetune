#!/usr/bin/env python3
"""
Dynamic Fine-tuning script for Qwen3 model using Unsloth
Configurable via YAML files for different hardware setups and datasets
"""

import os
import gc
import yaml
import argparse
import torch
import traceback
from pathlib import Path
from transformers import EarlyStoppingCallback
from typing import Dict, Any, Optional, List
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

class FineTuningConfig:
    """Configuration manager for fine-tuning parameters"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.validate_config()
        self.convert_types()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def convert_types(self):
        """Convert string values to appropriate types"""
        # Convert common numeric fields that might be strings
        training = self.config.get('training', {})
        
        # Convert learning rate and other float fields
        float_fields = ['learning_rate', 'warmup_ratio']
        for field in float_fields:
            if field in training and isinstance(training[field], str):
                try:
                    training[field] = float(training[field])
                except ValueError:
                    pass
        
        # Convert integer fields
        int_fields = ['per_device_train_batch_size', 'gradient_accumulation_steps', 
                     'num_train_epochs', 'logging_steps', 'warmup_steps', 
                     'save_total_limit', 'save_steps', 'dataset_num_proc']
        for field in int_fields:
            if field in training and isinstance(training[field], str):
                try:
                    training[field] = int(training[field])
                except ValueError:
                    pass
        
        # Convert model fields
        model = self.config.get('model', {})
        model_int_fields = ['max_seq_length']
        for field in model_int_fields:
            if field in model and isinstance(model[field], str):
                try:
                    model[field] = int(model[field])
                except ValueError:
                    pass
        
        # Convert LoRA fields
        lora = model.get('lora', {})
        lora_int_fields = ['r', 'alpha', 'random_state']
        lora_float_fields = ['dropout']
        for field in lora_int_fields:
            if field in lora and isinstance(lora[field], str):
                try:
                    lora[field] = int(lora[field])
                except ValueError:
                    pass
        for field in lora_float_fields:
            if field in lora and isinstance(lora[field], str):
                try:
                    lora[field] = float(lora[field])
                except ValueError:
                    pass
    
    def validate_config(self):
        """Validate required configuration keys"""
        required_keys = ['model', 'training', 'dataset', 'hardware', 'output']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.config['model']
    
    def get_training_config(self) -> Dict[str, Any]:
        return self.config['training']
    
    def get_dataset_config(self) -> Dict[str, Any]:
        return self.config['dataset']
    
    def get_hardware_config(self) -> Dict[str, Any]:
        return self.config['hardware']
    
    def get_output_config(self) -> Dict[str, Any]:
        return self.config['output']

def setup_environment(hardware_config: Dict[str, Any]):
    """Set up environment variables for optimal training"""
    env_vars = hardware_config.get('environment_variables', {})
    
    # Default environment variables
    default_env = {
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        'CUDA_LAUNCH_BLOCKING': '1',
        'TORCH_USE_CUDA_DSA': '1',
        'TOKENIZERS_PARALLELISM': 'false'
    }
    
    # Merge with user-defined env vars
    for key, value in {**default_env, **env_vars}.items():
        os.environ[key] = str(value)

class LossBasedEarlyStoppingCallback:
    """Custom callback for early stopping based on training loss trends"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, 
                 min_steps: int = 100, check_interval: int = 10):
        self.patience = patience
        self.min_delta = min_delta
        self.min_steps = min_steps
        self.check_interval = check_interval
        self.loss_history: List[float] = []
        self.best_loss = float('inf')
        self.wait_count = 0
        self.stopped_early = False
    
    def on_log(self, logs: Dict[str, float], step: int):
        """Check if we should stop based on loss trends"""
        if step < self.min_steps:
            return False
            
        if 'loss' in logs and step % self.check_interval == 0:
            current_loss = logs['loss']
            self.loss_history.append(current_loss)
            
            # Keep only recent history
            if len(self.loss_history) > self.patience * 2:
                self.loss_history = self.loss_history[-self.patience * 2:]
            
            # Check for improvement
            if current_loss < (self.best_loss - self.min_delta):
                self.best_loss = current_loss
                self.wait_count = 0
                print(f"🔥 New best loss: {current_loss:.4f}")
            else:
                self.wait_count += 1
                print(f"⏳ No improvement for {self.wait_count}/{self.patience} checks")
            
            # Stop if no improvement for too long
            if self.wait_count >= self.patience:
                print(f"\n🛑 Early stopping triggered!")
                print(f"📊 Best loss: {self.best_loss:.4f}")
                print(f"⏱️ No improvement for {self.patience} checks")
                self.stopped_early = True
                return True
                
        return False

class SmartTrainingCallback:
    """Advanced callback with multiple stopping conditions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loss_callback = None
        self.target_loss = config.get('target_loss', None)
        self.max_time_minutes = config.get('max_time_minutes', None)
        self.start_time = None
        
        # Initialize loss-based early stopping if configured
        if config.get('enable_loss_early_stopping', False):
            self.loss_callback = LossBasedEarlyStoppingCallback(
                patience=config.get('early_stop_patience', 5),
                min_delta=config.get('early_stop_min_delta', 0.001),
                min_steps=config.get('early_stop_min_steps', 100),
                check_interval=config.get('early_stop_check_interval', 10)
            )
    
    def on_train_begin(self):
        import time
        self.start_time = time.time()
        print("🚀 Smart training callback initialized")
        if self.target_loss:
            print(f"🎯 Target loss: {self.target_loss}")
        if self.max_time_minutes:
            print(f"⏰ Max training time: {self.max_time_minutes} minutes")
    
    def should_stop(self, logs: Dict[str, float], step: int) -> tuple[bool, str]:
        """Check all stopping conditions"""
        import time
        
        # Check target loss
        if self.target_loss and 'loss' in logs:
            if logs['loss'] <= self.target_loss:
                return True, f"🎯 Target loss {self.target_loss} achieved! Current: {logs['loss']:.4f}"
        
        # Check time limit
        if self.max_time_minutes and self.start_time:
            elapsed_minutes = (time.time() - self.start_time) / 60
            if elapsed_minutes >= self.max_time_minutes:
                return True, f"⏰ Time limit reached: {elapsed_minutes:.1f}/{self.max_time_minutes} minutes"
        
        # Check loss-based early stopping
        if self.loss_callback and self.loss_callback.on_log(logs, step):
            return True, "📉 Loss-based early stopping triggered"
        
def print_gpu_memory(prefix: str = ""):
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix}GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")

def formatting_prompts_func(examples, format_type: str = "chatml"):
    """Format conversations for training with different formats"""
    texts = []
    
    if format_type == "chatml":
        for conversation in examples["messages"]:
            text = ""
            for message in conversation:
                role = message["role"]
                content = message["content"]
                text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            texts.append(text)
    
    elif format_type == "alpaca":
        for conversation in examples["messages"]:
            # Assume first message is instruction, second is response
            if len(conversation) >= 2:
                instruction = conversation[0]["content"]
                response = conversation[1]["content"]
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}\n"
                texts.append(text)
    
    elif format_type == "simple_qa":
        # For simple Q&A format
        for conversation in examples.get("conversations", examples.get("messages", [])):
            if isinstance(conversation, dict) and "question" in conversation and "answer" in conversation:
                text = f"Question: {conversation['question']}\nAnswer: {conversation['answer']}\n"
                texts.append(text)
    
    return {"text": texts}

def load_and_prepare_model(model_config: Dict[str, Any], hardware_config: Dict[str, Any]):
    """Load model with quantization and add LoRA adapters"""
    print(f"Loading {model_config['name']}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config['name'],
        max_seq_length=model_config.get('max_seq_length', 2048),
        dtype=model_config.get('dtype', None),
        load_in_4bit=model_config.get('load_in_4bit', True),
        trust_remote_code=model_config.get('trust_remote_code', True),
        use_cache=model_config.get('use_cache', False),
        device_map=hardware_config.get('device_map', "auto"),
    )
    
    print("Adding LoRA adapters...")
    lora_config = model_config.get('lora', {})
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.get('r', 16),
        target_modules=lora_config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_alpha=lora_config.get('alpha', 16),
        lora_dropout=lora_config.get('dropout', 0),
        bias=lora_config.get('bias', "none"),
        use_gradient_checkpointing=lora_config.get('gradient_checkpointing', "unsloth"),
        random_state=lora_config.get('random_state', 42),
        use_rslora=lora_config.get('use_rslora', False),
        loftq_config=lora_config.get('loftq_config', None),
    )
    
    return model, tokenizer

def prepare_dataset(dataset_config: Dict[str, Any]):
    """Load and format the dataset"""
    dataset_path = dataset_config['path']
    dataset_format = dataset_config.get('format', 'json')
    
    if dataset_format == 'json':
        dataset = load_dataset("json", data_files=dataset_path)["train"]
    elif dataset_format == 'jsonl':
        dataset = load_dataset("json", data_files=dataset_path)["train"]
    elif dataset_format == 'csv':
        dataset = load_dataset("csv", data_files=dataset_path)["train"]
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
    
    print(f"Dataset size: {len(dataset)}")
    
    # Apply formatting to dataset
    format_type = dataset_config.get('conversation_format', 'chatml')
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, format_type), 
        batched=True
    )
    
    # Apply any filtering if specified
    if 'max_length' in dataset_config:
        max_length = dataset_config['max_length']
        dataset = dataset.filter(lambda example: len(example['text']) <= max_length)
    
    # Take subset if specified
    if 'subset_size' in dataset_config and dataset_config['subset_size'] > 0:
        subset_size = min(dataset_config['subset_size'], len(dataset))
        dataset = dataset.select(range(subset_size))
        print(f"Using subset of {subset_size} samples")
    
    # Verify the formatting worked
    print("Formatted sample:")
    print(dataset[0]["text"][:500] + "..." if len(dataset[0]["text"]) > 500 else dataset[0]["text"])
    
    return dataset

def create_training_config(training_config: Dict[str, Any], model_config: Dict[str, Any], 
                          output_dir: str) -> SFTConfig:
    """Create training configuration"""
    
    # Extract training parameters with defaults and ensure proper types
    batch_size = int(training_config.get('per_device_train_batch_size', 2))
    gradient_accumulation_steps = int(training_config.get('gradient_accumulation_steps', 4))
    num_epochs = int(training_config.get('num_train_epochs', 1))
    learning_rate = float(training_config.get('learning_rate', 2e-4))
    max_seq_length = int(model_config.get('max_seq_length', 2048))
    
    return SFTConfig(
        # Basic settings
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        
        # Memory optimization
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        gradient_checkpointing_kwargs=training_config.get('gradient_checkpointing_kwargs', 
                                                        {"use_reentrant": False}),
        optim=training_config.get('optimizer', "adamw_8bit"),
        
        # Precision settings
        bf16=training_config.get('bf16', True),
        fp16=training_config.get('fp16', False),
        
        # Logging
        logging_steps=int(training_config.get('logging_steps', 10)),
        logging_first_step=bool(training_config.get('logging_first_step', True)),
        output_dir=output_dir,
        report_to=training_config.get('report_to', "none"),
        
        # Scheduler settings
        warmup_steps=int(training_config.get('warmup_steps', 10)),
        warmup_ratio=float(training_config.get('warmup_ratio', 0.1)),
        lr_scheduler_type=training_config.get('lr_scheduler_type', "linear"),
        
        # Save settings
        save_strategy=training_config.get('save_strategy', "epoch"),
        save_total_limit=int(training_config.get('save_total_limit', 2)),
        save_steps=int(training_config.get('save_steps', 500)),
        
        # Dataset settings
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=training_config.get('packing', False),
        dataloader_pin_memory=training_config.get('dataloader_pin_memory', False),
        remove_unused_columns=training_config.get('remove_unused_columns', True),
        seed=training_config.get('seed', 42),
        
        # Additional optimizations
        dataset_num_proc=int(training_config.get('dataset_num_proc', 2)),
        per_device_eval_batch_size=int(training_config.get('per_device_eval_batch_size', batch_size)),
    )

def test_model(model, tokenizer, test_prompts: list):
    """Test the trained model with sample prompts"""
    print("\n🧪 Testing the model...")
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    for i, prompt in enumerate(test_prompts[:3]):  # Test first 3 prompts
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        
        messages = [{"role": "user", "content": prompt}]
        
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
            do_sample=True,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")

def save_model(model, tokenizer, output_config: Dict[str, Any], output_dir: str):
    """Save the trained model in specified format"""
    save_method = output_config.get('save_method', 'merged_16bit')
    
    print(f"\nSaving model using method: {save_method}")
    
    if save_method in ['merged_16bit', 'merged_4bit', 'lora']:
        model.save_pretrained_merged(
            output_dir, 
            tokenizer,
            save_method=save_method,
        )
    else:
        # Fallback to standard save
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Model saved to {output_dir}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Dynamic Fine-tuning with YAML config")
    parser.add_argument("--config", "-c", type=str, required=True, 
                       help="Path to YAML configuration file")
    parser.add_argument("--output-dir", "-o", type=str, 
                       help="Output directory (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    config_manager = FineTuningConfig(args.config)
    
    # Get configurations
    model_config = config_manager.get_model_config()
    training_config = config_manager.get_training_config()
    dataset_config = config_manager.get_dataset_config()
    hardware_config = config_manager.get_hardware_config()
    output_config = config_manager.get_output_config()
    
    # Set output directory
    output_dir = args.output_dir or output_config.get('directory', './finetuned-model')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup environment
    setup_environment(hardware_config)
    
    try:
        print("🚀 Starting dynamic fine-tuning...")
        print(f"📊 Config: {args.config}")
        print(f"📁 Output: {output_dir}")
        print(f"🔧 Model: {model_config['name']}")
        print(f"📚 Dataset: {dataset_config['path']}")
        
        # Load and prepare model
        model, tokenizer = load_and_prepare_model(model_config, hardware_config)
        
        # Prepare dataset
        dataset = prepare_dataset(dataset_config)
        
        # Clear memory before training
        torch.cuda.empty_cache()
        gc.collect()
        print_gpu_memory("Before training - ")
        
        # Create training configuration
        training_args = create_training_config(training_config, model_config, output_dir)
        
        # Clear memory again
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get smart training config
        smart_config = config_manager.config.get('smart_training', {})
        
        # Create smart trainer with early stopping capabilities
        trainer = create_smart_trainer(model, tokenizer, dataset, training_args, smart_config)
        
def create_smart_trainer(model, tokenizer, dataset, training_args, smart_config: Dict[str, Any]):
    """Create trainer with smart early stopping capabilities"""
    
    # Initialize callbacks
    callbacks = []
    smart_callback = SmartTrainingCallback(smart_config)
    
    # Add HuggingFace early stopping if evaluation is enabled
    if smart_config.get('enable_eval_early_stopping', False):
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=smart_config.get('eval_early_stop_patience', 3),
            early_stopping_threshold=smart_config.get('eval_early_stop_threshold', 0.001)
        ))
    
    # Create custom trainer class with smart stopping
    class SmartSFTTrainer(SFTTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.smart_callback = smart_callback
            self.smart_callback.on_train_begin()
        
        def log(self, logs: Dict[str, float]) -> None:
            """Override log method to check stopping conditions"""
            super().log(logs)
            
            # Check if we should stop
            should_stop, reason = self.smart_callback.should_stop(logs, self.state.global_step)
            if should_stop:
                print(f"\n{reason}")
                print("💾 Saving model before stopping...")
                self.save_model()
                print("✅ Model saved successfully")
                self.control.should_training_stop = True
    
    # Create the smart trainer
    trainer = SmartSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        callbacks=callbacks,
        dataset_num_proc=smart_config.get('dataset_num_proc', 2),
    )
    
    return trainer
        
        # Print GPU stats
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            print(f"🔧 GPU: {gpu_stats.name}")
            print(f"💾 Total Memory: {gpu_stats.total_memory / 1024**3:.1f} GB")
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"📊 Allocated before training: {allocated:.2f} GB")
            print(f"🆓 Free memory: {(gpu_stats.total_memory / 1024**3) - allocated:.2f} GB")
        
        # Train the model with early stopping handling
        print("\n🏋️ Training started...")
        print("💡 Press Ctrl+C to stop training early and save the model")
        
        try:
            trainer_stats = trainer.train()
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
            print("💾 Saving model at current checkpoint...")
            trainer.save_model()
            trainer_stats = None
        
        print("\n✅ Training complete!")
        if trainer_stats and hasattr(trainer_stats, 'training_loss'):
            print(f"📉 Final loss: {trainer_stats.training_loss:.4f}")
        else:
            print("📊 Training was stopped early")
        
        # Save the model
        save_model(model, tokenizer, output_config, output_dir)
        
        # Test the model if test prompts are provided
        test_prompts = output_config.get('test_prompts', [])
        if test_prompts:
            test_model(model, tokenizer, test_prompts)
        
        # Save training config for reference
        config_save_path = Path(output_dir) / "training_config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump(config_manager.config, f, default_flow_style=False, indent=2)
        print(f"📋 Training config saved to {config_save_path}")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ Out of memory error!")
        print_gpu_memory("OOM Error - ")
        print("\n💡 Memory-saving suggestions:")
        print("1. Reduce per_device_train_batch_size")
        print("2. Increase gradient_accumulation_steps") 
        print("3. Reduce max_seq_length")
        print("4. Enable packing in training config")
        print("5. Reduce LoRA rank (r parameter)")
        print("6. Use smaller model variant")
        torch.cuda.empty_cache()
        raise

    except Exception as e:
        print(f"\n❌ Training error: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()