import json
import hashlib

def create_deduplicated_jsonl(clean_file_path):
    """
    Sadece geçerli JSON satırları içeren bir dosyayı okur ve
    tekrar eden kayıtları temizleyerek yeni bir dosya oluşturur.
    """
    # Çıktı dosyasının ismini belirle
    base_name = clean_file_path.replace('.jsonl', '')
    deduplicated_file_path = f"{base_name}_tekrarsiz.jsonl"

    seen_hashes = set()
    written_count = 0
    duplicate_count = 0

    print(f"\n'{clean_file_path}' dosyasındaki tekrarlar temizleniyor...")

    try:
        with open(clean_file_path, 'r', encoding='utf-8') as f_in, \
             open(deduplicated_file_path, 'w', encoding='utf-8') as f_out:

            for line in f_in:
                if not line.strip():
                    continue

                # Her ihtimale karşı satırı tekrar parse ediyoruz
                data = json.loads(line)
                canonical_form = json.dumps(data, sort_keys=True)
                content_hash = hashlib.sha256(canonical_form.encode('utf-8')).hexdigest()

                if content_hash not in seen_hashes:
                    # Bu içeriği ilk defa görüyoruz
                    seen_hashes.add(content_hash)
                    f_out.write(line) # Orijinal satırı yaz
                    written_count += 1
                else:
                    # Bu bir tekrar, atla
                    duplicate_count += 1
        
        print("\nİşlem tamamlandı!")
        print(f"✔️ '{deduplicated_file_path}' dosyasına {written_count} adet benzersiz satır yazıldı.")
        print(f"🗑️ Toplam {duplicate_count} adet tekrarlanan satır atlandı.")


    except FileNotFoundError:
        print(f"Hata: '{clean_file_path}' adında bir dosya bulunamadı. Lütfen önce 1. Adımı çalıştırdığınızdan emin olun.")
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")


# --- KULLANIM ---
create_deduplicated_jsonl('/home/ika/llm/finetune/data/data_combined_temiz.jsonl')