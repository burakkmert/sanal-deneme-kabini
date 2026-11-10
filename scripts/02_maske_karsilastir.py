"""Adım 2: Orijinal maske ile akıllı maskeyi yan yana gösterir (görüntü üretmez).

Kullanım:
  python scripts/02_maske_karsilastir.py --image FOTO_YOLU
Çıktı: outputs/02_karsilastirma/<foto>/karsilastirma.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent
CATVTON_DIR = os.environ.get("CATVTON_DIR")
if not CATVTON_DIR or not Path(CATVTON_DIR, "model", "cloth_masker.py").exists():
    sys.exit("CATVTON_DIR ayarlı değil. Önce .\\ortam.ps1 çalıştır.")
sys.path.insert(0, CATVTON_DIR)
sys.path.insert(0, str(ROOT))   # 'kabin' paketini bulabilmek için

from huggingface_hub import snapshot_download  # noqa: E402
from model.cloth_masker import AutoMasker  # noqa: E402
from utils import resize_and_crop  # noqa: E402

from kabin.masking import ParseMaps, build_upper_mask  # noqa: E402
from kabin.kayit import numarali_klasor  # noqa: E402


def diziye_cevir(x, boyut):
    img = x if isinstance(x, Image.Image) else Image.fromarray(np.asarray(x).astype(np.uint8))
    if img.size != boyut:
        img = img.resize(boyut, Image.NEAREST)
    arr = np.array(img)
    return arr[..., 0] if arr.ndim == 3 else arr


def boya(kisi, maske, renk):
    """Maskelenen bölgeyi yarı saydam renkle gösterir."""
    img = np.array(kisi).astype(np.float32)
    m = np.array(maske) > 127
    img[m] = img[m] * 0.45 + np.array(renk, np.float32) * 0.55
    return Image.fromarray(img.astype(np.uint8))


def yan_yana(resimler, basliklar):
    w, h = resimler[0].size
    tuval = Image.new("RGB", (w * len(resimler) + 6 * (len(resimler) - 1), h + 30), "white")
    ciz = ImageDraw.Draw(tuval)
    for i, (r, b) in enumerate(zip(resimler, basliklar)):
        x = i * (w + 6)
        tuval.paste(r, (x, 30))
        ciz.text((x + 8, 8), b, fill="black")
    return tuval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--grow_px", type=int, default=12)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    out_dir = numarali_klasor(ROOT / "outputs" / "02_karsilastirma", Path(args.image).stem)

    repo_path = snapshot_download(repo_id="zhengchong/CatVTON")
    masker = AutoMasker(densepose_ckpt=os.path.join(repo_path, "DensePose"),
                        schp_ckpt=os.path.join(repo_path, "SCHP"), device=args.device)

    kisi = resize_and_crop(Image.open(args.image).convert("RGB"), (768, 1024))
    sonuc = masker(kisi, "upper")
    orijinal = sonuc["mask"].convert("L")
    maps = ParseMaps(*(diziye_cevir(sonuc[k], kisi.size) for k in ("densepose", "schp_lip", "schp_atr")))

    resimler = [boya(kisi, orijinal, (60, 120, 255))]
    basliklar = [f"ORIJINAL  %{(np.array(orijinal) > 127).mean() * 100:.1f}"]
    orijinal.save(out_dir / "maske_orijinal.png")
    for kol, ad in (("long", "uzun kol"), ("short", "kisa kol"), ("sleeveless", "kolsuz")):
        m = Image.fromarray(build_upper_mask(maps, sleeve=kol, grow_px=args.grow_px)["mask"])
        m.save(out_dir / f"maske_akilli_{kol}.png")
        resimler.append(boya(kisi, m, (255, 60, 60)))
        basliklar.append(f"AKILLI {ad}  %{(np.array(m) > 127).mean() * 100:.1f}")

    yan_yana(resimler, basliklar).save(out_dir / "karsilastirma.png")
    print("Maske kapsaması (görüntünün yüzdesi):")
    for b in basliklar:
        print("  ", b)
    print(f"Kaydedildi: {out_dir / 'karsilastirma.png'}")


if __name__ == "__main__":
    main()