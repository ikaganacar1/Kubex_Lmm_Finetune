import pandas as pd
import asyncio
import json
import re
from googletrans import Translator

# Load the dataset
df = pd.read_parquet("hf://datasets/sidddddddddddd/kubernetes-with-ood/data/train-00000-of-00001.parquet")

async def translate_text(text):
    """Translate text to Turkish"""
    try:
        translator = Translator()
        print(f"Attempting to translate: {text[:100]}...")
        result = await translator.translate(text, dest='tr')
        print(f"Translation successful!")
        print(f"Original language detected: {result.src}")
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        print(f"Error type: {type(e)}")
        # Return original text if translation fails
        return text

def replace_devops_with_kubex(text):
    """Replace DevOps with KUBEX (case insensitive)"""
    return re.sub(r'\bDevOps\b', 'KUBEX', text, flags=re.IGNORECASE)

def parse_instruction_format(text):
    """Parse the <s>[INST] ... [/INST] ... </s> format"""
    # Remove <s> and </s> tags first
    text = text.strip()
    if text.startswith('<s>'):
        text = text[3:]
    if text.endswith('</s>'):
        text = text[:-4]
    text = text.strip()
    
    # Pattern to match [INST] or [Inst] ... [/INST] or [/Inst] ... format (case insensitive)
    # Using case-insensitive flag for better matching
    pattern = r'\[INST\]\s*(.*?)\s*\[/INST\]\s*(.*?)$'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        user_message = match.group(1).strip()
        assistant_message = match.group(2).strip()
        return user_message, assistant_message
    else:
        # If format doesn't match, return original text as user message
        print(f"Warning: Could not parse instruction format from: {text[:100]}...")
        return text, ""

def create_chatml_format(user_content, assistant_content):
    """Convert to ChatML format with KubeX system message"""
    system_message = "You are a Kubernetes management assistant. You have access to the KubeX API to manage clusters, deployments, pods, and other Kubernetes resources. Only use the provided API tools. If asked about topics outside Kubernetes management, politely decline."
    
    chatml = {
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user", 
                "content": user_content
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }
    
    return chatml

async def process_data_entry(text):
    """Process a single data entry through the pipeline"""
    print(f"Original: {text}")
    print("-" * 50)
    
    # Step 1: Translate to Turkish
    translated_text = await translate_text(text)
    print(f"Translated: {translated_text}")
    print("-" * 50)
    
    # Step 2: Replace DevOps with KUBEX
    kubex_text = replace_devops_with_kubex(translated_text)
    print(f"After KUBEX replacement: {kubex_text}")
    print("-" * 50)
    
    # Step 3: Parse instruction format
    user_msg, assistant_msg = parse_instruction_format(kubex_text)
    print(f"User message: {user_msg}")
    print(f"Assistant message: {assistant_msg}")
    print("-" * 50)
    
    # Step 4: Convert to ChatML format
    chatml_result = create_chatml_format(user_msg, assistant_msg)
    print("ChatML Format:")
    print(json.dumps(chatml_result, ensure_ascii=False, indent=2))
    
    return chatml_result

async def main():
    """Main processing function"""
    processed_count = 0
    failed_count = 0
    
    # Process first 10 entries for testing (change this to len(df["text"]) for full dataset)
    num_entries_to_process = min(1000, len(df["text"]))
    
    # Open file in append mode, but clear it first
    output_file = 'processed_kubex_dataset.jsonl'
    
    # Clear the file at the beginning
    with open(output_file, 'w+', encoding='utf-8') as f:
        pass  # Just clear the file
    
    # Now process and append each entry
    for i, text in enumerate(df["text"][:num_entries_to_process]):
        print(f"\n=== Processing Entry {i+1} ===")
        
        try:
            result = await process_data_entry(text)
            
            # Immediately write to file after each successful processing
            with open(output_file, 'a', encoding='utf-8') as f:
                json_line = json.dumps(result, ensure_ascii=False)
                f.write(json_line + '\n')
            
            processed_count += 1
            print(f"✓ Entry {i+1} saved to {output_file}")
            
        except Exception as e:
            print(f"✗ Error processing entry {i+1}: {e}")
            failed_count += 1
            # Continue processing other entries even if one fails
            continue
    
    print(f"\nAll processed entries saved to '{output_file}'")
    
    # Print summary statistics
    print("\n=== Processing Summary ===")
    print(f"Total entries attempted: {num_entries_to_process}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")

# Run the pipeline
if __name__ == "__main__":
    asyncio.run(main())