# OTÜ Olasılık Quizi

Ostim Teknik Üniversitesi MATH 204 Olasılık ve İstatistik dersi için hazırlanmış, Streamlit tabanlı **kişiye özel 5 soruluk quiz** uygulaması. Soruların sayısal değerleri öğrenci numarası ve quiz oturumundan deterministik olarak üretilir; böylece her öğrenci aynı soru tiplerini farklı sayılarla çözer.

## Özellikler

**Öğrenci tarafı**

- Ad soyad, 9 haneli öğrenci numarası ve öğretmen seçimiyle giriş.
- Her soru 20 puan; cevaplar mutlak tolerans ile (varsayılan ±0,05) otomatik puanlanır.
- Her oturumda her öğrenci yalnızca bir kez teslim yapabilir.
- Geri sayım sayacı; cevaplar değiştikçe taslak olarak kaydedilir ve süre dolduğunda son kaydedilen cevaplar otomatik gönderilir.
- İsteğe bağlı yorum aşaması: quiz bittikten sonra cevaplar kilitlenir, öğrenci her soru için çözüm açıklaması yazar.
- İsteğe bağlı sekme/uygulama değişikliği takibi: açıksa, quiz sırasında başka sekmeye geçen öğrencinin quizi sonlandırılır ve 0 puan kaydedilir.
- Soru metinlerinin kopyalanmasını ve OCR ile okunmasını zorlaştıran görsel önlemler.
- Türkçe / İngilizce arayüz (kenar çubuğundan veya `?lang=tr` / `?lang=en` URL parametresiyle).

**Öğretmen paneli** (kenar çubuğunda, öğretmen koduyla)

- Quizi açma/kapatma; başlangıç tarihi ve saati, süre (1, 3, 5, 10, 15, 20 veya 30 dk) ve cevap toleransı ayarı.
- 25 soru tipinden 5 soru kutusunu seçme (Bayes, toplam olasılık, PMF/CDF, beklenen değer ve varyans, binom, Poisson, geometrik, hipergeometrik, normal, üssel, gamma, Weibull vb.).
- Yorum aşamasını ve sekme değişikliği cezasını açıp kapatma.
- `PUBLIC_BASE_URL` tanımlıysa öğrenci giriş bağlantısı ve QR kodu.
- Oturum bazlı rapor (katılan sayısı, ortalama, en yüksek, en düşük puan), öğrenci arama ve kayıt tablosu.
- Panel yalnızca seçilen öğretmenin öğrencilerine ait kayıtları gösterir; seçili öğretmene ait kayıtları silme seçeneği. Tüm öğretmenler aynı öğretmen kodunu kullanır.

## Kurulum

Python 3.11 önerilir (`runtime.txt`).

```bash
git clone https://github.com/ufukasia/otu-quiz.git
cd otu-quiz
pip install -r requirements.txt
```

## Çalıştırma

```bash
streamlit run streamlit_app.py
```

Depoda bir `.devcontainer` yapılandırması da bulunur; GitHub Codespaces'ta açıldığında bağımlılıklar kurulur ve uygulama 8501 portunda başlatılır.

## Yapılandırma

Uygulama ayarları sırasıyla `st.secrets` (`.streamlit/secrets.toml` veya Streamlit Cloud → App Settings → Secrets), ortam değişkenleri ve proje klasöründeki `.env` dosyasından okur. Başlamak için `.env.example` dosyasını `.env` olarak kopyalayın:

```bash
cp .env.example .env
```

| Anahtar | Açıklama |
|---|---|
| `TEACHER_CODE_HASH` | Öğretmen kodunun SHA-256 özeti (64 karakter hex). Tanımlı değilse öğretmen paneli açılmaz. |
| `TEACHER_CODE` | Alternatif: düz metin öğretmen kodu (yalnızca `TEACHER_CODE_HASH` yoksa kullanılır). |
| `PUBLIC_BASE_URL` | Öğrencilere gösterilecek uygulama adresi; QR kodu bu adresten üretilir. |
| `APP_LANGUAGE` | Varsayılan arayüz dili: `tr` veya `en` (varsayılan `tr`). |
| `APP_TIMEZONE` | Saat dilimi (varsayılan `Europe/Istanbul`). |

Öğretmen kodunun özetini üretmek için:

```bash
python -c "import hashlib; print(hashlib.sha256('yeni-kod'.encode()).hexdigest())"
```

`.env` ve `.streamlit/secrets.toml` dosyaları `.gitignore` içindedir; gerçek kodları depoya eklemeyin.

Öğretmen listesi `streamlit_app.py` içindeki `TEACHER_OPTIONS` sabitinde, soru tipleri `question_bank.py` dosyasında tanımlıdır.

## Veri

Sonuçlar ve cevap taslakları `outputs/quiz_results.db` (SQLite) dosyasında, quiz ayarları `outputs/quiz_control.json` dosyasında tutulur. Veritabanı boşsa eski sürümlerden kalan `outputs/quiz_results.csv` ve `outputs/quiz_answer_drafts.csv` dosyaları içe aktarılır. Bu dosyalar öğrenci adı ve numarası içerir; gerçek sınav verilerini herkese açık bir depoya göndermeyin.

## Testler

```bash
python -m unittest discover -s tests
```

## Proje yapısı

```
otu-quiz/
├── streamlit_app.py            # Uygulama (öğrenci ekranı + öğretmen paneli)
├── question_bank.py            # 25 soru tipi ve kişiye özel değer üretimi
├── components/tab_monitor/     # Sekme değişikliği takibi için Streamlit bileşeni
├── tests/test_question_bank.py # Soru bankası testleri
├── outputs/                    # Quiz ayarları ve sonuç dosyaları
├── *.tex                       # Haftalık ders sunumlarının LaTeX kaynakları
├── _fix_tab_js.py              # Yardımcı bakım betiği
├── .streamlit/config.toml      # Tema ayarları
├── .env.example                # Örnek yapılandırma
├── requirements.txt
└── runtime.txt
```

`.tex` sunumları `lutbeamer` belge sınıfını kullanır; bu sınıf dosyası depoda bulunmadığından sunumlar tek başına derlenmez.

## Lisans

Apache License 2.0. Ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.

## İletişim

Dr. Öğr. Üyesi Ufuk Asil, Ostim Teknik Üniversitesi. Sorular ve öneriler için GitHub üzerinden issue açabilirsiniz.
