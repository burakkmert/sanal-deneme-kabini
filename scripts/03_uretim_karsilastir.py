"""Adım 3: Aynı kişi + kıyafet + seed ile iki üretim: orijinal maske vs akıllı maske.

Kullanım:
  python scripts/03_uretim_karsilastir.py --kisi KISI_FOTO --kiyafet KIYAFET_FOTO 
Ayar verilmezse kol, bolluk ve etek payı kıyafet fotoğrafından tahmin edilir.
Elle vermek için: --kol long --bolluk bol --etek 40
Çıktı: outputs/03_uretim/NNN_<kisi>_<kol>_<bolluk>_etek<N>/karsilastirma.png + tek tek görseller + ayarlar.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # 'kabin' paketini bulabilmek için

from kabin.catvton import Motor  # noqa: E402
from kabin.kayit import numarali_klasor  # noqa: E402
from kabin.masking import build_upper_mask  # noqa: E402


def boya(kisi, maske, renk):
    img = np.array(kisi).astype(np.float32)
    m = np.array(maske) > 127
    img[m] = img[m] * 0.45 + np.array(renk, np.float32) * 0.55
    return Image.fromarray(img.astype(np.uint8))


def yan_yana(resimler, basliklar):
    w, h = resimler[0].size
    tuval = Image.new("RGB", (w * len(resimler) + 6 * (len(resimler) - 1), h + 30), "white")
    ciz = ImageDraw.Draw(tuval)
    for i, (r, b) in enumerate(zip(resimler, basliklar)):
        tuval.paste(r, (i * (w + 6), 30))
        ciz.text((i * (w + 6) + 8, 8), b, fill="black")
    return tuval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kisi", required=True, help="kişi fotoğrafı")
    ap.add_argument("--kiyafet", required=True, help="kıyafet fotoğrafı")
    ap.add_argument("--kol", default=None, choices=["long", "short", "sleeveless"],
                    help="YENİ kıyafetin kol boyu (verilmezse kıyafet fotoğrafından tahmin edilir)")
    ap.add_argument("--adim", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bolluk", default=None, choices=["dar", "normal", "bol"],
                    help="YENİ kıyafetin bolluğu (verilmezse tahmin edilir)")
    ap.add_argument("--etek", type=int, default=None,
                    help="etek payı (piksel); verilmezse kıyafetin boyundan tahmin edilir")
    ap.add_argument("--boyun", type=float, default=0.6,
                    help="boynun ne kadarı yakaya açılsın: 0 = hiç, 1 = çeneye kadar")
    ap.add_argument("--grow_px", type=int, default=None, help="kenar payını elle ver (deney için)")
    ap.add_argument("--maskeleyici", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    # Verilmeyen ayarlar kıyafet fotoğrafından tahmin edilir (CLIP, işlemcide çalışır
    # ki ekran kartı belleği üretime kalsın).
    tahmin = None
    if args.kol is None or args.bolluk is None or args.etek is None:
        from kabin.garment_clip import KiyafetCLIP, maske_ayarlari
        c = KiyafetCLIP(device="cpu").classify(Image.open(args.kiyafet))
        tahmin = maske_ayarlari(c)
        args.kol = args.kol or tahmin["sleeve"]
        args.bolluk = args.bolluk or {"tight": "dar", "normal": "normal", "loose": "bol"}[tahmin["fit"]]
        args.etek = tahmin["hem_px"] if args.etek is None else args.etek
        print(f"Kıyafet analizi: kol {c['sleeve_p']}  bolluk {c['fit']}  boy {c['length']}")
        if not tahmin["emin"]:
            print("  UYARI: kol tahmini emin değil, gerekirse --kol ile elle ver.")
    print(f"Kullanılan ayarlar: kol={args.kol} bolluk={args.bolluk} etek={args.etek}")

    motor = Motor(maskeleyici_cihaz=args.maskeleyici)
    kisi, kiyafet = motor.hazirla(Image.open(args.kisi), Image.open(args.kiyafet))

    orijinal, haritalar = motor.maskele(kisi, "upper")
    fit = {"dar": "tight", "normal": "normal", "bol": "loose"}[args.bolluk]
    akilli = Image.fromarray(build_upper_mask(
        haritalar, sleeve=args.kol, fit=fit, hem_px=args.etek, neck=args.boyun, grow_px=args.grow_px)["mask"])
    maskeler = {"orijinal": motor.yumusat(orijinal), "akilli": motor.yumusat(akilli)}

    klasor = numarali_klasor(ROOT / "outputs" / "03_uretim", f"{Path(args.kisi).stem}_{args.kol}_{args.bolluk}_etek{args.etek}")
    kisi.save(klasor / "kisi.png")
    kiyafet.save(klasor / "kiyafet.png")

    sonuclar, bilgiler = {}, {}
    for ad, maske in maskeler.items():
        print(f"Üretiliyor: {ad} maske ({args.adim} adım)...")
        sonuclar[ad], bilgiler[ad] = motor.uret(kisi, kiyafet, maske, args.adim, args.cfg, args.seed)
        maske.save(klasor / f"maske_{ad}.png")
        sonuclar[ad].save(klasor / f"sonuc_{ad}.png")
        print(f"  {bilgiler[ad]}")

    yan_yana(
        [kisi, kiyafet, boya(kisi, maskeler["orijinal"], (60, 120, 255)), sonuclar["orijinal"],
         boya(kisi, maskeler["akilli"], (255, 60, 60)), sonuclar["akilli"]],
        ["KISI", "KIYAFET", "ORIJINAL MASKE", "ORIJINAL SONUC", "AKILLI MASKE", "AKILLI SONUC"],
    ).save(klasor / "karsilastirma.png")

    ayarlar = {"kisi": str(args.kisi), "kiyafet": str(args.kiyafet), "kol": args.kol,
               "bolluk": args.bolluk, "etek": args.etek,
               "otomatik": tahmin is not None, "boyun": args.boyun, "grow_px": args.grow_px,
               "adim": args.adim, "cfg": args.cfg, "seed": args.seed,
               "sonuc": bilgiler}
    (klasor / "ayarlar.json").write_text(json.dumps(ayarlar, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Kaydedildi: {klasor / 'karsilastirma.png'}")


if __name__ == "__main__":
    main()