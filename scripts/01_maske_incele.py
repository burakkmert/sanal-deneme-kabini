"""Adım 1: CatVTON'un AutoMasker'ı bize ne veriyor?

AutoMasker maske üretirken üç segmentasyon haritası kullanıyor:
  densepose : vücut bölgeleri (gövde, üst kol, ön kol, el, yüz...)
  schp_lip  : kıyafet/vücut ayrıştırma (üst giysi, ceket, pantolon, saç...)
  schp_atr  : başka bir veri setiyle eğitilmiş ikinci ayrıştırıcı
Bu script bunları kaydeder, böylece akıllı maskeyi neyin üstüne kuracağımızı görürüz.

Kullanım:
  python scripts/01_maske_incele.py --image FOTO_YOLU --cloth_type upper
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# CatVTON kodunu bulabilmek için onun klasörünü Python yoluna ekliyoruz
CATVTON_DIR = os.environ.get("CATVTON_DIR")
if not CATVTON_DIR or not Path(CATVTON_DIR, "model", "cloth_masker.py").exists():
    sys.exit("CATVTON_DIR ayarlı değil ya da yanlış. Örnek: $env:CATVTON_DIR='C:\\...\\tryon'")
sys.path.insert(0, CATVTON_DIR)

from huggingface_hub import snapshot_download  # noqa: E402
from model.cloth_masker import AutoMasker, vis_mask  # noqa: E402
from utils import resize_and_crop  # noqa: E402

LIP_NAMES = ["arka plan", "şapka", "saç", "eldiven", "güneş gözlüğü", "üst giysi",
             "elbise", "ceket", "çorap", "pantolon", "tulum", "atkı", "etek", "yüz",
             "sol kol", "sağ kol", "sol bacak", "sağ bacak", "sol ayakkabı", "sağ ayakkabı"]


def renklendir(etiketler: np.ndarray) -> Image.Image:
    """Her etiket numarasına sabit bir renk verir (0 = siyah arka plan)."""
    rng = np.random.default_rng(7)
    palet = rng.integers(40, 255, size=(256, 3), dtype=np.uint8)
    palet[0] = 0
    return Image.fromarray(palet[etiketler])


def diziye_cevir(x, boyut):
    if x is None:
        return None
    img = x if isinstance(x, Image.Image) else Image.fromarray(np.asarray(x).astype(np.uint8))
    if img.size != boyut:
        img = img.resize(boyut, Image.NEAREST)
    arr = np.array(img)
    return arr[..., 0] if arr.ndim == 3 else arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="kişi fotoğrafı")
    ap.add_argument("--cloth_type", default="upper", choices=["upper", "lower", "overall", "inner", "outer"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    out_dir = Path("outputs") / "01_inceleme" / f"{Path(args.image).stem}_{args.cloth_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_path = snapshot_download(repo_id="zhengchong/CatVTON")  # zaten indirili, sadece yolu verir
    masker = AutoMasker(densepose_ckpt=os.path.join(repo_path, "DensePose"),
                        schp_ckpt=os.path.join(repo_path, "SCHP"), device=args.device)

    kisi = resize_and_crop(Image.open(args.image).convert("RGB"), (768, 1024))
    sonuc = masker(kisi, args.cloth_type)

    print("\nAutoMasker'ın döndürdüğü anahtarlar:", list(sonuc.keys()))
    kisi.save(out_dir / "kisi.png")
    sonuc["mask"].save(out_dir / "maske_orijinal.png")
    vis_mask(kisi, sonuc["mask"]).save(out_dir / "maske_ustunde.png")

    for ad in ("densepose", "schp_lip", "schp_atr"):
        arr = diziye_cevir(sonuc.get(ad), kisi.size)
        if arr is None:
            print(f"  {ad}: YOK")
            continue
        renklendir(arr).save(out_dir / f"harita_{ad}.png")
        etiketler = sorted(np.unique(arr).tolist())
        print(f"  {ad}: boyut={arr.shape}, etiketler={etiketler}")
        if ad == "schp_lip":
            for e in etiketler:
                if e < len(LIP_NAMES):
                    print(f"      {e:2d} = {LIP_NAMES[e]:14s} %{(arr == e).mean() * 100:5.1f}")

    print(f"\nKaydedildi: {out_dir.resolve()}")


if __name__ == "__main__":
    main()