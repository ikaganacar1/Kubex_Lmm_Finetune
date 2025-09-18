import json
import os
from typing import List, Dict
from datetime import datetime

class ChatMLToHarmonyConverter:
    def __init__(self):
        # Ana sistem promptu - HER ZAMAN Türkçe
        self.system_prompt = """Sen Kubernetes ve DevOps konularında uzmanlaşmış yardımcı bir yapay zeka asistanısın. Konteyner orkestrasyonu, dağıtım stratejileri ve bulut-native teknolojiler hakkında doğru ve pratik rehberlik sağlıyorsun.

ÖNEMLI: Her zaman Türkçe cevap ver. İngilizce sorular gelirse de cevabını Türkçe yaz. Teknik terimler için gerekirse parantez içinde İngilizce karşılığını verebilirsin.

Bilgi kesim tarihi: 2024-06
Güncel tarih: 2025-06-28

Mantık yürütme: yüksek

# Geçerli kanallar: analysis, commentary, final. Her mesaj için kanal belirtilmelidir."""
    
    def detect_language(self, text: str) -> str:
        """Detect if the text is Turkish or English"""
        turkish_chars = set('çğıöşüÇĞIİÖŞÜ')
        turkish_words = {'nedir', 'nasıl', 'neden', 'hangi', 'kubernetes', 'pod', 'service', 'deployment'}
        
        # Check for Turkish characters
        if any(char in text for char in turkish_chars):
            return 'tr'
        
        # Check for Turkish words
        text_lower = text.lower()
        turkish_word_count = sum(1 for word in turkish_words if word in text_lower)
        
        if turkish_word_count > 0:
            return 'tr'
        
        return 'en'
    
    def convert_single_conversation(self, chatml_conversation: Dict) -> str:
        """Convert a single ChatML conversation to Harmony format - Always Turkish"""
        
        messages = chatml_conversation.get('messages', [])
        if not messages:
            return ""
        
        # HER ZAMAN Türkçe sistem promptu kullan
        harmony_conversation = f"<|start|>system<|message|>{self.system_prompt}<|end|>\n\n"
        
        # Process each message
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '').strip()
            
            if not content:
                continue
            
            if role == 'user':
                harmony_conversation += f"<|start|>user<|message|>{content}<|end|><|start|>assistant<|message|>"
            elif role == 'assistant':
                harmony_conversation += f"{content}<|end|>\n\n"
        
        # Remove trailing newlines and ensure proper ending
        harmony_conversation = harmony_conversation.rstrip()
        if not harmony_conversation.endswith('<|end|>'):
            harmony_conversation += '<|end|>'
        
        return harmony_conversation
    
    def convert_jsonl_file(self, input_file: str, output_file: str) -> int:
        """Convert entire JSONL file from ChatML to Harmony format - Always Turkish"""
        
        print(f"🔄 Converting {input_file} to Harmony format...")
        print(f"📝 Output will be saved to: {output_file}")
        print("🇹🇷 ALL conversations will use Turkish system prompt (model will always respond in Turkish)")
        
        converted_count = 0
        skipped_count = 0
        
        try:
            with open(input_file, 'r', encoding='utf-8') as infile:
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    
                    for line_num, line in enumerate(infile, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            # Parse ChatML conversation
                            chatml_conv = json.loads(line)
                            
                            # Convert to Harmony format (always Turkish)
                            harmony_conv = self.convert_single_conversation(chatml_conv)
                            
                            if harmony_conv:
                                outfile.write(harmony_conv + '\n\n')
                                converted_count += 1
                            else:
                                skipped_count += 1
                                print(f"  ⚠️  Skipped empty conversation on line {line_num}")
                            
                            # Progress update
                            if line_num % 100 == 0:
                                print(f"  📊 Processed {line_num} lines, converted {converted_count}, skipped {skipped_count}")
                                
                        except json.JSONDecodeError as e:
                            skipped_count += 1
                            print(f"  ❌ JSON error on line {line_num}: {e}")
                        except Exception as e:
                            skipped_count += 1
                            print(f"  ❌ Error on line {line_num}: {e}")
        
        except FileNotFoundError:
            print(f"❌ Input file not found: {input_file}")
            return 0
        except Exception as e:
            print(f"❌ File processing error: {e}")
            return 0
        
        print(f"\n✅ Conversion completed!")
        print(f"📊 Results:")
        print(f"   - Successfully converted: {converted_count} conversations")
        print(f"   - Skipped: {skipped_count} conversations")
        print(f"   - Output file: {output_file}")
        print(f"🇹🇷 Model will ALWAYS respond in Turkish regardless of input language")
        
        return converted_count
    
    def preview_harmony_format(self, output_file: str, num_examples: int = 2):
        """Preview the converted Harmony format"""
        
        print(f"\n{'='*70}")
        print(f"HARMONY FORMAT PREVIEW")
        print(f"{'='*70}")
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                conversations = content.split('\n\n\n')  # Split by double newlines
                
                for i, conv in enumerate(conversations[:num_examples]):
                    if conv.strip():
                        print(f"\n📝 CONVERSATION {i+1}:")
                        print(f"{'─'*60}")
                        print(conv.strip())
                        print(f"{'─'*60}")
                        
        except Exception as e:
            print(f"❌ Preview error: {e}")
    
    def batch_convert_multiple_files(self, file_configs: List[Dict]):
        """Convert multiple files - All with Turkish system prompt"""
        
        print("🚀 Batch conversion starting...")
        print("🇹🇷 ALL models will be configured to respond in Turkish")
        print("=" * 60)
        
        total_converted = 0
        
        for i, config in enumerate(file_configs, 1):
            input_file = config['input']
            output_file = config['output']
            
            print(f"\n📁 File {i}/{len(file_configs)}: {input_file}")
            
            if not os.path.exists(input_file):
                print(f"   ❌ File not found, skipping...")
                continue
            
            converted = self.convert_jsonl_file(
                input_file=input_file,
                output_file=output_file
            )
            
            total_converted += converted
        
        print(f"\n🎉 Batch conversion completed!")
        print(f"📊 Total conversations converted: {total_converted}")
        print(f"🇹🇷 All models configured to ALWAYS respond in Turkish!")

def main():
    """Main function for converting datasets"""
    
    print("🔄 ChatML to OpenAI Harmony Format Converter")
    print("🇹🇷 MODEL HER ZAMAN TÜRKÇE KONUŞACAK")
    print("=" * 60)
    
    converter = ChatMLToHarmonyConverter()
    
    # Configuration for multiple datasets - All will use Turkish
    file_configs = [
        {
            'input': 'kubernetes-docs-Concepts.jsonl',
            'output': 'kubernetes-docs-Concepts-harmony.txt'
        },
        {
            'input': 'kubernetes-docs-Tasks.jsonl', 
            'output': 'kubernetes-docs-Tasks-harmony.txt'
        },
    ]
    
    # Batch convert all files (all will be Turkish)
    converter.batch_convert_multiple_files(file_configs)
    
    # Preview results from first available file
    for config in file_configs:
        if os.path.exists(config['output']):
            print(f"\n🔍 Preview from {config['output']}:")
            converter.preview_harmony_format(config['output'], num_examples=1)
            break

if __name__ == "__main__":
    main()