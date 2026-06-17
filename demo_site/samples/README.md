# Örnek videolar (samples)

Bu klasör, demo arayüzünde "hazır örnek" olarak listelenen videolar içindir.

> **Not:** Projenin geliştirilmesinde **FakeAVCeleb v1.2** veri setinden klipler kullanıldı.
> FakeAVCeleb yalnızca araştırma amaçlı lisanslı olduğu ve yeniden dağıtımı yasak olduğu için
> (ayrıca gerçek kişilerin deepfake'lerini içerdiğinden) **örnek videolar bu repoya dahil edilmemiştir.**
> Bu nedenle `.mp4`/`.avi`/`.mov`/`.mkv` dosyaları `.gitignore` ile hariç tutulur.

## Kendi örneklerinizi nasıl eklersiniz?

`.mp4` dosyalarınızı doğrudan bu klasöre kopyalayın. Demo, başlatıldığında klasörü tarar
ve dosyaları otomatik listeler. Uygulama ayrıca **kendi videonuzu yükleyerek** de çalışır
(samples olmasa bile demo tam çalışır).

### Ground-truth (doğru etiket) için isimlendirme

`inference.py` içindeki `category_from_path`, dosya adındaki anahtar kelimeye göre
gerçek etiketi çıkarır. Otomatik karşılaştırma istiyorsanız dosya adına şu ibarelerden
birini ekleyin:

| Dosya adında geçen | Kategori | Video | Ses |
|---|---|---|---|
| `RealVideo-RealAudio` | R-R | gerçek | gerçek |
| `FakeVideo-RealAudio` | F-R | sahte | gerçek |
| `RealVideo-FakeAudio` | R-F | gerçek | sahte |
| `FakeVideo-FakeAudio` | F-F | sahte | sahte |

Örnek: `RealVideo-FakeAudio__ornek1.mp4`

İsimde bu ibareler yoksa video yine analiz edilir, sadece "doğru/yanlış" karşılaştırması yapılmaz.
