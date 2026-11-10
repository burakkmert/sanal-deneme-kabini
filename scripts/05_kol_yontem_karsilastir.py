"""Adım 5: Kol boyu tahmininde iki yöntemi aynı fotoğraflarda karşılaştırır.

  kural : garment.py  (silüet + geometri)
  clip  : garment_clip.py (CLIP sıfır-örnek sınıflandırma)

Dosya adları gerçek etiketle başlamalı: uzun_..., kısa_... (ya da kisa_...), kolsuz_...
  python scripts/05_kol_yontem_karsilastir.py --yol KLASOR
İlk çalıştırmada CLIP modeli (~600 MB) Hugging Face'ten iner. CatVTON gerekmez.
"""
import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # 'kabin' paketini bulabilmek için

from kabin.garment import classify_sleeve, measure  # noqa: E402
from kabin.garment_clip import KiyafetCLIP  # noqa: E402
from kabin.kayit import numarali_klasor  # noqa: E402

TR = {"long": "uzun", "short": "kısa", "sleeveless": "kolsuz", None: "?"}


def gercek_etiket(ad):
    ad = ad.lower()
    if ad.startswith("uzun"):
        return "long"
    if ad.startswith("kolsuz"):
        return "sleeveless"
    if ad.startswith(("kısa", "kisa")):
        return "short"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yol", required=True, help="kıyafet fotoğrafları klasörü")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    args = ap.parse_args()

    dosyalar = sorted(p for p in Path(args.yol).iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if not dosyalar:
        sys.exit("Fotoğraf bulunamadı.")
    clip = KiyafetCLIP(args.model)
    print(f"CLIP cihazı: {clip.device}\n")

    satirlar, dogru = [], {"kural": 0, "clip": 0}
    etiketli = 0
    print(f"{'dosya':24s} {'gerçek':>7s} {'kural':>10s} {'clip':>8s}  clip olasılıkları      bolluk  boy")
    for d in dosyalar:
        resim = Image.open(d).convert("RGB")
        gercek = gercek_etiket(d.name)
        m = measure(resim)
        kural = classify_sleeve(m) if m is not None else None
        c = clip.classify(resim)
        if gercek:
            etiketli += 1
            dogru["kural"] += kural == gercek
            dogru["clip"] += c["sleeve"] == gercek
        isaret = lambda t: "" if not gercek else ("✓" if t == gercek else "✗")  # noqa: E731
        p = c["sleeve_p"]
        print(f"{d.name[:24]:24s} {TR[gercek]:>7s} {TR[kural] if kural else 'ölçülemedi':>8s} {isaret(kural):1s} "
              f"{TR[c['sleeve']]:>6s} {isaret(c['sleeve']):1s}  "
              f"u{p['long']:.2f} k{p['short']:.2f} s{p['sleeveless']:.2f}   {c['fit']:6s}  {c['length']}")
        satirlar.append({"dosya": d.name, "gercek": gercek, "kural": kural, "clip": c["sleeve"],
                         "p_uzun": p["long"], "p_kisa": p["short"], "p_kolsuz": p["sleeveless"],
                         "clip_bolluk": c["fit"], "clip_boy": c["length"]})

    klasor = numarali_klasor(ROOT / "outputs" / "05_kol_yontem", Path(args.yol).name)
    with open(klasor / "sonuclar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    if etiketli:
        print(f"\nKol boyu doğruluğu ({etiketli} etiketli fotoğraf):")
        print(f"  kural tabanlı : {dogru['kural']}/{etiketli}")
        print(f"  CLIP          : {dogru['clip']}/{etiketli}")
    print(f"Kaydedildi: {klasor}")


if __name__ == "__main__":
    main()