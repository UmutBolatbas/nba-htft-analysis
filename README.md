# NBA İlk Yarı / Maç Sonucu Analizi

NBA maçlarında **1/2** (ev sahibi devreyi önde kapatır, deplasman kazanır) ve **2/1** (tersi) sonuçlarının maç öncesi oranlardan önceden kestirilip kestirilemeyeceğini inceleyen istatistiksel analiz betikleri.

Sonuçlar normal süreye göre değerlendirilir (uzatmalar sayılmaz). Böylece 9 seçenekli İY/MS marketiyle (1/1, 1/X, 1/2, X/1, X/X, X/2, 2/1, 2/X, 2/2) birebir karşılaştırma yapılabilir.

## Yöntem

- **Brown hareketi modeli** (Stern, 1994): Maç öncesi handikaptan, devre farkı ile normal süre farkının ortak normal dağılımı kurulur ve 9 İY/MS seçeneğinin olasılıkları hesaplanır.
- **Walk-forward doğrulama:** Her test dönemi yalnızca kendinden önceki verilerle eğitilir; model gelecekteki sonuçları görmez.
- **Ek bilgi testleri:** Bir oran alanının (barem, handikap, marj, oran basamakları, bahis yüzdeleri vb.) modelin ötesinde bilgi taşıyıp taşımadığı olabilirlik oranı testiyle ölçülür.
- **Çoklu test düzeltmesi:** Benjamini–Hochberg yöntemi kullanılır. Bulunan desenler ayrı bir dönemde yeniden doğrulanır (keşif/doğrulama ayrımı).
- **Sezon analizi:** Fiyatlamanın sezondan sezona değişip değişmediği ve bir dönemde bulunan etkilerin sonraki döneme taşınıp taşınmadığı incelenir.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `backtest.py` | Veri yükleme ve temel walk-forward backtest |
| `htft_deep.py` | Brown modeli, kalibrasyon, ek bilgi testleri, adil oran tablosu |
| `htft_signals.py` | Oran alanlarının 1/2 ve 2/1 için sinyal testleri |
| `htft_digits.py` | Oran basamaklarına dayalı desen testleri |
| `htft_seasons.py` | Sezonlar arası farklar ve sezon içi kararlılık |
| `htft_iddaa.py` | İddaa İY/MS oranlarıyla marj, kalibrasyon, strateji ve oran yapısı testleri |
| `htft_iddaa_seasons.py` | İddaa fiyatlamasının sezon dinamiği ve uyarlanan model |
| `htft_iddaa_1122.py` | 1/1 ve 2/2 oranlarının 1/2 ve 2/1 için bilgi taşıyıp taşımadığı |
| `teams.py` | Takım adı → kısaltma eşlemesi |

## Veri

Veri dosyaları ve veri toplama betikleri bu depoda **yer almaz**. Betikler `data/` klasöründe şu dosyaları bekler:

- `data/nba_betmgm_closing.csv`: maç başına kapanış oranları (maç sonucu, handikap, alt/üst) ve çeyrek skorları
- `data/raw/mackolik/matches.csv`, `data/raw/mackolik/odds.csv`: İddaa oranları (uzun format) ve maç skorları

## Çalıştırma

```bash
pip install -r requirements.txt
python htft_deep.py
python htft_iddaa.py        # data/nba_iddaa.csv dosyasını üretir; iddaa_* betiklerinden önce çalıştırılmalı
python htft_iddaa_seasons.py
```

## Not

Bu depo istatistiksel bir araştırmadır, bahis tavsiyesi değildir.
