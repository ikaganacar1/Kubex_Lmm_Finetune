import json

def separate_valid_and_invalid_json(original_file_path):
    """
    Bir JSONL dosyasını okur ve içeriği geçerli ve geçersiz olarak
    iki ayrı dosyaya yazar.
    """
    # Çıktı dosyalarının isimlerini orijinal isme göre belirle
    base_name = original_file_path.replace('.jsonl', '')
    clean_file_path = f"{base_name}_temiz.jsonl"
    error_file_path = f"{base_name}_hatali.txt"

    valid_count = 0
    invalid_count = 0

    print(f"'{original_file_path}' dosyası işleniyor...")

    try:
        # İki yeni dosyayı yazma modunda aç
        with open(original_file_path, 'r', encoding='utf-8') as f_in, \
             open(clean_file_path, 'w', encoding='utf-8') as f_clean, \
             open(error_file_path, 'w', encoding='utf-8') as f_error:

            for line_number, line in enumerate(f_in, 1):
                # Satır boşsa atla
                if not line.strip():
                    continue
                
                try:
                    # JSON'ı parse etmeyi dene
                    json.loads(line)
                    # Başarılı olursa, temiz dosyaya yaz
                    f_clean.write(line)
                    valid_count += 1
                except json.JSONDecodeError:
                    # Hata olursa, hatalı satırlar dosyasına yaz
                    f_error.write(f"Satır {line_number}: {line}")
                    invalid_count += 1

        print("\nİşlem tamamlandı!")
        print(f"✔️ '{clean_file_path}' dosyasına {valid_count} adet geçerli satır yazıldı.")
        print(f"❌ '{error_file_path}' dosyasına {invalid_count} adet hatalı satır yazıldı.")
        print("\nLütfen hatalı satırlar dosyasını inceleyerek veri kaybı olup olmadığını kontrol edin.")

    except FileNotFoundError:
        print(f"Hata: '{original_file_path}' adında bir dosya bulunamadı.")
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")

# --- KULLANIM ---
# Kendi dosya adınızı buraya yazın
separate_valid_and_invalid_json('/home/ika/llm/finetune/data/data_combined.jsonl')