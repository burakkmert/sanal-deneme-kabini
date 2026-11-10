"""Adım 4: Kıyafet fotoğraflarının şeklini ölçer (GPU ve CatVTON gerekmez).

Tek fotoğraf ya da bir klasör verilebilir:
  python scripts/04_kiyafet_olc.py --yol KIYAFET_FOTO_ya_da_KLASOR
Çıktı: outputs/04_kiyafet/NNN_<ad>/  ->  her kıyafet için silüet görseli + olcumler.csv
"""
import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # 'kabin' paketini bulabilmek için

from kabin.garment import classify_sleeve, debug_image, measure  # noqa: E402
from kabin.kayit import numarali_klasor  # noqa: E402

SUTUNLAR = ["dosya", "tahmin", "sleeve_area", "sleeve_reach", "aspect", "width_ratio", "torso_w", "body_len"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yol", required=True, help="kıyafet fotoğrafı ya da klasörü")
    args = ap.parse_args()

    yol = Path(args.yol)
    dosyalar = [yol] if yol.is_file() else sorted(
        p for p in yol.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if not dosyalar:
        sys.exit("Fotoğraf bulunamadı.")

    klasor = numarali_klasor(ROOT / "outputs" / "04_kiyafet", yol.stem)
    satirlar = []
    print(f"{'dosya':34s} {'tahmin':>11s} {'kol alanı':>10s} {'kol erişimi':>12s} {'boy/gen':>8s}")
    for d in dosyalar:
        resim = Image.open(d).convert("RGB")
        m = measure(resim)
        if m is None:
            print(f"{d.name[:34]:34s}   ölçülemedi (arka plan düz değil ya da silüet bulunamadı)")
            satirlar.append({"dosya": d.name})
            continue
        m["tahmin"] = classify_sleeve(m)
        debug_image(resim, m).save(klasor / f"{d.stem}.png")
        satirlar.append({"dosya": d.name, **{k: m[k] for k in SUTUNLAR[1:]}})
        print(f"{d.name[:34]:34s} {m['tahmin']:>11s} {m['sleeve_area']:10.3f} "
              f"{m['sleeve_reach']:12.2f} {m['aspect']:8.2f}")

    with open(klasor / "olcumler.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUTUNLAR)
        w.writeheader()
        w.writerows(satirlar)
    print(f"Kaydedildi: {klasor}")


if __name__ == "__main__":
    main()