import json
from collections import defaultdict

# Terminalde renkli çıktı için basit bir sınıf (isteğe bağlı, ama okunaklılığı artırır)
class Renk:
    HEADER = '\033[95m'
    MAVI = '\033[94m'
    YESIL = '\033[92m'
    SARI = '\033[93m'
    KIRMIZI = '\033[91m'
    ENDC = '\033[0m' # Rengi sıfırla
    BOLD = '\033[1m'

def pretty_check_duplicates_in_jsonl(file_path):
    """
    Bir JSONL dosyasındaki tekrarlanan satırları bulur ve sonucu
    görsel olarak zenginleştirilmiş bir formatta raporlar.
    """
    print(f"{Renk.BOLD}Dosya işleniyor: {Renk.MAVI}{file_path}{Renk.ENDC}\n")

    # görülen_ilk_satir: {içerik: ilk görüldüğü satır no}
    # tekrarlar: {içerik: [tekrarlandığı satır no'ları]}
    gorulen_ilk_satir = {}
    tekrarlar = defaultdict(list)
    invalid_line_count = 0
    line_number = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_number += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    # İçeriği standart hale getir (anahtarları sırala)
                    canonical_form = json.dumps(data, sort_keys=True)

                    if canonical_form in gorulen_ilk_satir:
                        # Bu içerik daha önce görüldü, yani bu bir tekrar.
                        tekrarlar[canonical_form].append(line_number)
                    else:
                        # Bu içeriği ilk defa görüyoruz.
                        gorulen_ilk_satir[canonical_form] = line_number

                except json.JSONDecodeError:
                    print(f"{Renk.SARI}Uyarı:{Renk.ENDC} Satır {line_number} geçerli bir JSON değil ve atlandı.")
                    invalid_line_count += 1
        
        # --- Raporlama Kısmı ---
        if tekrarlar:
            print(f"{Renk.KIRMIZI}{Renk.BOLD}--- Tekrarlanan Kayıtlar Bulundu ---\n{Renk.ENDC}")
            
            # Toplam tekrar sayısını hesapla
            total_duplicate_entries = sum(len(lines) for lines in tekrarlar.values())

            for i, (form, lines) in enumerate(tekrarlar.items()):
                original_line_num = gorulen_ilk_satir[form]
                
                # Tekrarlanan JSON içeriğini güzel formatta yazdır
                pretty_json = json.dumps(json.loads(form), indent=2, ensure_ascii=False)
                
                print(f"{Renk.BOLD}Tekrar #{i+1}{Renk.ENDC}")
                print(f"{Renk.MAVI}{pretty_json}{Renk.ENDC}")
                print(f"  -> Bu içerik ilk olarak {Renk.YESIL}{original_line_num}. satırda{Renk.ENDC} görüldü.")
                print(f"  -> Ve şu satırlarda tekrar etti: {Renk.KIRMIZI}{lines}{Renk.ENDC}\n")
        
        else:
            print(f"{Renk.YESIL}--- Sonuç: Dosyada tekrarlanan içerik bulunamadı. ---\n{Renk.ENDC}")

        # --- Genel Özet ---
        print(f"{Renk.HEADER}{Renk.BOLD}--- Dosya Özeti ---\n{Renk.ENDC}")
        print(f"Toplam Okunan Satır Sayısı : {line_number}")
        print(f"Geçersiz JSON Satır Sayısı: {invalid_line_count}")
        print(f"Benzersiz İçerik Sayısı  : {Renk.YESIL}{len(gorulen_ilk_satir)}{Renk.ENDC}")
        if tekrarlar:
             print(f"Tekrarlanan İçerik Türü  : {Renk.KIRMIZI}{len(tekrarlar)}{Renk.ENDC}")
             print(f"Toplam Tekrar Kaydı Sayısı: {Renk.KIRMIZI}{total_duplicate_entries}{Renk.ENDC} (Orijinalleri hariç)")
        print("\n" + "="*40)


    except FileNotFoundError:
        print(f"{Renk.KIRMIZI}Hata: '{file_path}' adında bir dosya bulunamadı.{Renk.ENDC}")
    except Exception as e:
        print(f"{Renk.KIRMIZI}Beklenmedik bir hata oluştu: {e}{Renk.ENDC}")


# --- KULLANIM ÖRNEĞİ ---
pretty_check_duplicates_in_jsonl('data_combined.jsonl')