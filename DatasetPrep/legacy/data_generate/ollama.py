import os
import json
import time
import requests
import re


# Ollama server configuration
ollama_server = "http://10.150.96.44:11434"
model_name = "qwen3-coder:30b"  # Updated model name

output_file = 'SimpleSingleStepOperations.jsonl'
num_examples_to_generate = 8000 # Set the total number of examples you want
examples_per_batch = 5  # Generate 5 examples per API call

prompt = """Elbette, bir önceki prompt'u güncelleyerek verdiğiniz tüm endpoint listesini, yapay zekanın başvuracağı temel kaynak olarak ekliyorum. Bu sayede, AI'nın sadece geçerli `operationId`'leri kullanması ve bu listeden çeşitli seçimler yapması sağlanacaktır.

İşte en güncel ve kapsamlı prompt versiyonu:

-----

### **Yeniden Düzenlenmiş ve Geliştirilmiş Prompt (v3)**

**Hedef:** KubeX Kubernetes operasyonları için **5 adet**, birbirinden tamamen **farklı, yaratıcı ve Türkçe** çok aşamalı eğitim verisi oluştur.

**EN ÖNEMLİ KURALLAR:**

1.  **Tamamen Türkçe İçerik:** Üretilecek tüm içerik (`user` mesajı, `assistant` mesajları, `tool` cevabından üretilen özet) istisnasız olarak **tamamen Türkçe** olmalıdır. Dil kullanımı doğal ve akıcı olsun.
2.  **Maksimum Çeşitlilik:** Her bir örnek hem senaryo hem de kullanılan dil açısından bir diğerinden belirgin şekilde farklı olmalıdır. Aşağıdaki çeşitlilik kurallarına ve sağlanan API listesine mutlaka uy.
3.  **Kesin JSON Formatı:** Yanıtın **sadece 5 adet** JSON nesnesi içermelidir. Her JSON nesnesi ayrı bir satırda olmalı, markdown veya kod bloğu içinde olmamalıdır.

**Çıktı Formatı:**

```
{"messages": [...]}
{"messages": [...]}
{"messages": [...]}
{"messages": [...]}
{"messages": [...]}
```

-----

### **JSON Yapısı ve İçerik Gereksinimleri:**

1.  **Anahtar:** Her JSON nesnesi sadece `"messages"` anahtarına sahip olmalıdır.
2.  **Rol Sıralaması:** `"messages"` dizisi şu 5 rolü tam olarak bu sırayla içermelidir: `system`, `user`, `assistant`, `tool`, `assistant`.
3.  **Rol İçerikleri:**
      * `system`: Bu mesaj sabit kalacak: "Sen KubeX Kubernetes yönetim asistanısın. KubeX API'sini kullanarak cluster, deployment, pod, sanal makine ve diğer Kubernetes kaynaklarını yönetebilirsin. Sadece sana sunulan KubeX API araçlarını kullan. Kubernetes yönetimi dışındaki konularda sorulan soruları kibarca reddet."
      * `user`: **Yaratıcı ve çeşitli** kullanıcı istekleri oluştur. (Ayrıntılar aşağıda)
      * `assistant` (ilk cevap): Kullanıcının isteğini anladığını belirten kısa, doğal bir Türkçe cümle kur ve **TEK BİR** KubeX API araç çağrısı yap. Araç çağrısı yaparken aşağıdaki listede verilen geçerli `operationId`'lerden birini kullan.
      * `tool`: KubeX API'sinden dönen gerçekçi bir JSON yanıtını string formatında sun. **Veriler çeşitli ve gerçekçi olsun** (örneğin, her şey "Ready" veya "Running" olmak zorunda değil).
      * `assistant` (son cevap): `tool` içeriğini baz alarak kullanıcıya **anlaşılır, iyi formatlanmış (Markdown kullanarak) ve tamamen Türkçe** bir özet sun.

-----

### **Çeşitlilik ve Yaratıcılık Kuralları:**

  * **Farklı Kullanıcı Profilleri Oluştur:** Acemi kullanıcı ("Pod'ları göster"), deneyimli mühendis ("staging-cluster'daki 'monitoring' namespace'i için pod özetini getir") veya aceleci yönetici ("Prod cluster özeti lütfen") gibi farklı profillerde sorular oluştur.
  * **Farklı Senaryolar Kurgula:** Sadece listeleme yapma. Bir **sağlık kontrolü**, bir kaynağın **detaylı bilgisi**, bir **ölçekleme** (`scale`) işlemi, bir **log** (`logs`) sorgusu veya bir **sanal makine başlatma** (`start`) gibi farklı API komutlarını her örnekte değiştirerek kullan.
  * **Gerçekçi API Yanıtları Simüle Et:** `tool` rolünün içeriğini çeşitlendir. Bir node `Ready` iken diğeri `NotReady` olabilir. Bir pod `Running` iken başkası `Pending` veya `CrashLoopBackOff` durumunda olabilir.

-----

### **Kullanılabilecek API Operasyonları (`operationId` listesi):**

Aşağıdaki listeden her örnek için **farklı bir tane** `operationId` seçmeye özen göster. Bu liste, kullanabileceğin geçerli araçları tanımlar.

  * **Cluster Yönetimi:** `cluster_index`, `cluster_create`, `cluster_update`, `cluster_summary`, `cluster_toggle_favorite`, `cluster_health`
  * **Kubeconfig Yönetimi:** `kubeconfig_index`, `kubeconfig_create`
  * **İş (Job) Yönetimi:** `job_index`, `job_create`, `job_run`
  * **Namespace Yönetimi:** `namespace_show`, `namespace_pod_summary`, `namespace_index`
  * **Node Yönetimi:** `node_show`, `node_index`
  * **Deployment Yönetimi:** `deployment_index`, `deployment_show`, `deployment_scale`, `deployment_redeploy`, `deployment_get_detailed_config`, `deployment_update_container_image`, `deployment_update_image_pull_secret`, `deployment_update_environment`, `deployment_update_labels`, `deployment_get_pods_by_deployment_id_real_time`
  * **Service Yönetimi:** `service_index`, `service_show`, `service_index_port_forwarding_list`, `service_start_port_forward`, `service_delete_port_forward`
  * **İzleme ve Uyarılar:** `prometheus_proxy`, `alert_get`
  * **Pod Yönetimi:** `pod_index`, `pod_logs`
  * **Depolama (Storage) Yönetimi:** `pvc_index`, `pvc_delete`, `storage_class_index`
  * **Sanal Makine (VM) Yönetimi:** `virtual_machine_index`, `virtual_machine_instances`, `virtual_machine_sets`, `virtual_machine_start`, `virtual_machine_stop`, `virtual_machine_create`, `virtual_machine_vnc`, `virtual_machine_stop_vnc`, `virtual_machine_instant_screenshot`, `virtual_machine_vnc_keepalive`
  * **Veri ve İmaj Yönetimi (VM):** `data_volume_index_all_namespaces`, `data_volume_index_single_namespace`, `data_volume_create`, `data_volume_delete`, `data_volume_create_blank`, `image_index_all_namespaces`, `image_index_single_namespace`
  * **ISO Yönetimi:** `iso_index`, `iso_create`, `iso_show`, `iso_delete`, `iso_update`, `iso_upload_image`, `iso_set_image_pvc`
  * **Helm & Repository Yönetimi:** `repository_index`, `repository_add`, `repository_delete`, `repository_update`, `repository_health_of_helm`, `repository_install`
  * **Uygulama ve Ağ Yönetimi:** `installed_app_index`, `network_index`
  * **Diğer Araçlar:** `visionary_run`

-----

### **Örnek (Sadece Yapıyı Gösterir, İçeriği Kopyalama\!):**

Bu örnek sadece uyman gereken **yapıyı** göstermektedir. İçeriği, senaryoyu ve dili birebir kopyalama. Kendi yaratıcı ve çeşitli 5 örneğini bu yapıya göre oluştur.

````json
{"messages": [{"role": "system", "content": "Sen KubeX Kubernetes yönetim asistanısın. KubeX API'sini kullanarak cluster, deployment, pod, sanal makine ve diğer Kubernetes kaynaklarını yönetebilirsin. Sadece sana sunulan KubeX API araçlarını kullan. Kubernetes yönetimi dışındaki konularda sorulan soruları kibarca reddet."}, {"role": "user", "content": "Staging cluster'daki `api-gateway-7b...` podunun loglarına bakmam lazım, bir hata var sanırım."}, {"role": "assistant", "content": "Hemen `api-gateway-7b...` podunun loglarını getiriyorum.", "tool_calls": [{"type": "function", "function": {"name": "pod_logs", "arguments": "{\"cluster_id\": \"staging-k8s\", \"namespace\": \"default\", \"pod_name\": \"api-gateway-7b...\"}"}}]}, {"role": "tool", "content": "{\"logs\": \"time=\\\"2023-10-27T10:30:00Z\\\" level=info msg=\\\"Starting server on port 8080\\\"\\ntime=\\\"2023-10-27T10:31:15Z\\\" level=error msg=\\\"Failed to connect to upstream service: redis-service.default.svc.cluster.local\\\"\\ntime=\\\"2023-10-27T10:31:15Z\\\" level=fatal msg=\\\"Upstream connection error, shutting down...\\\"\"}"}, {"role": "assistant", "content": "İstediğiniz pod'un logları aşağıda belirtilmiştir:\n\n### `api-gateway-7b...` Pod Logları\n\n```\ntime=\"2023-10-27T10:30:00Z\" level=info msg=\"Starting server on port 8080\"\ntime=\"2023-10-27T10:31:15Z\" level=error msg=\"Failed to connect to upstream service: redis-service.default.svc.cluster.local\"\ntime=\"2023-10-27T10:31:15Z\" level=fatal msg=\"Upstream connection error, shutting down...\"\n```\n\nGörünüşe göre pod, `redis-service` isimli servise bağlanamadığı için kritik bir hata alıp kapanıyor."}]}
````

**Şimdi, yukarıdaki tüm kurallara ve çeşitlilik prensiplerine uyarak 5 yeni ve özgün örnek oluştur.**"""

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