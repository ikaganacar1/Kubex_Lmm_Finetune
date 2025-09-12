import os
import json
import time
import requests
import re


# Ollama server configuration
ollama_server = "http://10.150.96.44:11434"
model_name = "qwen3-coder:30b"  # Updated model name

output_file = 'ParameterRequired.jsonl'
num_examples_to_generate = 80000 # Set the total number of examples you want
examples_per_batch = 20 # Generate 5 examples per API call

prompt = """KubeX API'nin parametre gerektiren operasyonları için SADECE ChatML JSONL formatında eğitim verisi üret.

KATLI KURALLAR:
- Her satır tek JSON objesi
- Her JSON'da TAM 5 messages elementi (system->user->assistant->tool->assistant)
- Teknik terimler hariç %100 Türkçe
- Her operasyon minimum 3-4 parametre
- Tool response gerçekçi JSON
- Boşluk yok, sıkıştırılmış format

SADECE BU ENDPOINTleri KULLAN:
1. deployment_scale - replica sayısını değiştir
2. deployment_update_container_image - container imajı güncelle  
3. deployment_update_environment - ortam değişkenleri güncelle
4. deployment_update_labels - etiketleri güncelle
5. virtual_machine_start - VM başlat
6. virtual_machine_stop - VM durdur
7. virtual_machine_create - yeni VM oluştur
8. service_start_port_forward - port yönlendirme başlat
9. service_delete_port_forward - port yönlendirme durdur
10. pvc_delete - PVC sil
11. data_volume_create - data volume oluştur
12. data_volume_delete - data volume sil
13. data_volume_create_blank - boş data volume oluştur
14. repository_add - helm repo ekle
15. repository_install - helm chart kur
16. repository_delete - helm repo sil
17. cluster_create - yeni cluster oluştur
18. cluster_update - cluster güncelle
19. iso_upload_image - ISO imaj yükle
20. job_run - job çalıştır

ZORUNLU PARAMETRE ÖRNEKLERİ:
- "nginx deployment'ını production cluster'da backend namespace'inde 8 replica'ya ölçekle"
- "web-server VM'ini dev-cluster'da başlat"
- "mysql-pvc'yi staging cluster'dan kube-system namespace'inden sil"
- "postgres imajını v13.2'ye güncelle production'da api namespace'inde"
- "redis için localhost:6379'dan cluster'a port yönlendirme başlat"

TEMPLATE (ÖRNEKTEN BAĞIMSIZ EŞSİZ BİR ÇIKTI ÜRETMELSİN):
{"messages":[{"role":"system","content":"Sen KubeX Kubernetes yönetim asistanısın. KubeX API'sini kullanarak cluster, deployment, pod, sanal makine ve diğer Kubernetes kaynaklarını yönetebilirsin. Sadece sana sunulan KubeX API araçlarını kullan. Kubernetes yönetimi dışındaki konularda sorulan soruları kibarca reddet."},{"role":"user","content":"[PARAMETRE_İÇEREN_TALEP]"},{"role":"assistant","content":"[İŞLEM_AÇIKLAMASI]","tool_calls":[{"type":"function","function":{"name":"[API_FONKSIYON]","arguments":"{\"param1\":\"value1\",\"param2\":\"value2\",\"param3\":\"value3\"}"}}]},{"role":"tool","content":"{\"success\":true,\"message\":\"[REALISTIC_RESPONSE]\"}"},{"role":"assistant","content":"[SONUÇ_AÇIKLAMASI]"}]}

JSONL ÇIKTI - AÇIKLAMA YOK:"""

def call_ollama_api(prompt, max_retries=3, timeout=120):
    """
    Make an API call to Ollama server with retry logic
    Increased timeout for generating 5 examples
    """
    url = f"{ollama_server}/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    for attempt in range(max_retries):
        try:
            print(f"  -> Making API call to Ollama server (attempt {attempt + 1}/{max_retries})...")
            
            response = requests.post(
                url, 
                json=payload, 
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                print(f"  -> ⚠️ API call failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"  -> ⚠️ Request timed out (attempt {attempt + 1}/{max_retries})")
            
        except requests.exceptions.ConnectionError:
            print(f"  -> ⚠️ Connection error (attempt {attempt + 1}/{max_retries})")
            
        except Exception as e:
            print(f"  -> ⚠️ Unexpected error: {e} (attempt {attempt + 1}/{max_retries})")
        
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"  -> Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    raise Exception(f"Failed to get response from Ollama after {max_retries} attempts")

def parse_multiple_json_objects(response_text):
    """
    Parse multiple JSON objects from the response text.
    Returns a list of valid JSON objects.
    """
    # Clean up the response
    clean_text = response_text.strip()
    
    # Remove markdown code blocks if present
    clean_text = re.sub(r'^```(?:json)?\s*\n?', '', clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r'\n?```\s*$', '', clean_text, flags=re.MULTILINE)
    
    # Split by lines and try to parse each line as JSON
    lines = clean_text.split('\n')
    json_objects = []
    current_json = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # If line starts with {, it might be a new JSON object
        if line.startswith('{'):
            # If we have accumulated JSON, try to parse it first
            if current_json:
                try:
                    data = json.loads(current_json)
                    if 'messages' in data:
                        json_objects.append(data)
                except json.JSONDecodeError:
                    pass
            current_json = line
        else:
            # Continue building the current JSON object
            current_json += " " + line if current_json else line
    
    # Don't forget the last JSON object
    if current_json:
        try:
            data = json.loads(current_json)
            if 'messages' in data:
                json_objects.append(data)
        except json.JSONDecodeError:
            pass
    
    # Alternative approach: try to find JSON objects using regex
    if len(json_objects) < examples_per_batch:
        # Look for complete JSON objects in the text
        json_pattern = r'\{"messages":\s*\[.*?\]\}'
        matches = re.findall(json_pattern, clean_text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                if data not in json_objects:  # Avoid duplicates
                    json_objects.append(data)
            except json.JSONDecodeError:
                continue
    
    return json_objects

def test_ollama_connection():
    """
    Test connection to Ollama server
    """
    try:
        print("Testing connection to Ollama server...")
        url = f"{ollama_server}/api/tags"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Connection successful! Available models: {[m['name'] for m in models.get('models', [])]}")
            
            # Check if our target model is available
            model_names = [m['name'] for m in models.get('models', [])]
            if model_name not in model_names:
                print(f"⚠️ WARNING: Model '{model_name}' not found in available models.")
                print(f"Available models: {model_names}")
                
            return True
        else:
            print(f"❌ Connection failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def generate_data():
    """
    Main function to generate synthetic data and write it to a file.
    Now generates 5 examples per API call.
    """
    # Test connection first
    if not test_ollama_connection():
        print("🛑 HATA: Ollama sunucusuna bağlanılamıyor. Sunucu URL'sini kontrol edin ve tekrar deneyin.")
        return

    # Calculate how many batches we need
    total_batches = (num_examples_to_generate + examples_per_batch - 1) // examples_per_batch
    print(f"Starting generation of {num_examples_to_generate} examples in {total_batches} batches of {examples_per_batch} examples each...")
    print(f"Output file: '{output_file}'")

    examples_generated = 0
    
    # Open the file in append mode ('a') to add new lines without deleting existing ones
    with open(output_file, 'a', encoding='utf-8') as f:
        for batch_num in range(total_batches):
            try:
                remaining_examples = min(examples_per_batch, num_examples_to_generate - examples_generated)
                print(f"\nGenerating batch {batch_num + 1}/{total_batches} ({remaining_examples} examples)...")
                
                # Make the API call with retry logic
                response_text = call_ollama_api(prompt)
                
                # Parse multiple JSON objects from the response
                json_objects = parse_multiple_json_objects(response_text)
                
                if not json_objects:
                    print(f"  -> ⚠️ WARNING: No valid JSON objects found in batch {batch_num + 1}. Skipping.")
                    continue
                
                # Write each valid JSON object to the file
                successful_writes = 0
                for i, data in enumerate(json_objects[:remaining_examples]):  # Limit to remaining examples needed
                    try:
                        # Validate the structure
                        if 'messages' not in data:
                            print(f"  -> ⚠️ WARNING: JSON object {i+1} missing 'messages' key. Skipping.")
                            continue
                        
                        # Convert to compact JSON string (one line)
                        json_line = json.dumps(data, ensure_ascii=False)
                        
                        # Write the single-line JSON to the file
                        f.write(json_line + '\n')
                        successful_writes += 1
                        examples_generated += 1
                        
                    except Exception as e:
                        print(f"  -> ⚠️ WARNING: Error processing JSON object {i+1}: {e}")
                
                f.flush()  # Ensure data is written to disk immediately
                
                print(f"  -> ✅ Successfully wrote {successful_writes}/{len(json_objects)} examples from batch {batch_num + 1}")
                
                # Check if we've generated enough examples
                if examples_generated >= num_examples_to_generate:
                    break
                
            except Exception as e:
                print(f"  -> 🛑 An unexpected error occurred in batch {batch_num + 1}: {e}. Skipping batch.")
                
            # Small delay to avoid overwhelming the server
            time.sleep(0.2)

    print(f"\n✅ Generation complete. Generated {examples_generated} examples and saved to '{output_file}'.")

if __name__ == "__main__":
    generate_data()