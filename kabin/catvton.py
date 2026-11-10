"""CatVTON köprüsü: modeli ve maskeleyiciyi yükler, üretimi çalıştırır.

CatVTON kodu bu repoda DEĞİL. Yeri CATVTON_DIR ortam değişkeninden okunur
(ortam.ps1 bunu ayarlar). Yükleme ayarları CatVTON'un kendi app.py dosyasıyla aynı.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

CATVTON_DIR = os.environ.get("CATVTON_DIR")
if not CATVTON_DIR or not Path(CATVTON_DIR, "model", "cloth_masker.py").exists():
    raise RuntimeError("CATVTON_DIR ayarlı değil ya da yanlış. Önce .\\ortam.ps1 çalıştır.")
if CATVTON_DIR not in sys.path:
    sys.path.insert(0, CATVTON_DIR)

import torch  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from model.cloth_masker import AutoMasker  # noqa: E402
from model.pipeline import CatVTONPipeline  # noqa: E402
from utils import init_weight_dtype, resize_and_crop, resize_and_padding  # noqa: E402

from .masking import ParseMaps  # noqa: E402


def _diziye_cevir(x, boyut):
    """AutoMasker haritasını (H, W) etiket dizisine çevirir."""
    img = x if isinstance(x, Image.Image) else Image.fromarray(np.asarray(x).astype(np.uint8))
    if img.size != boyut:
        img = img.resize(boyut, Image.NEAREST)
    arr = np.array(img)
    return arr[..., 0] if arr.ndim == 3 else arr


class Motor:
    """Modeller bir kez yüklenir, sonra istenildiği kadar üretim yapılır."""

    def __init__(self, genislik=768, yukseklik=1024, maskeleyici_cihaz="cuda", sadece_maske=False):
        self.boyut = (genislik, yukseklik)
        repo = snapshot_download(repo_id="zhengchong/CatVTON")   # zaten indirili
        self.maskeleyici = AutoMasker(
            densepose_ckpt=os.path.join(repo, "DensePose"),
            schp_ckpt=os.path.join(repo, "SCHP"),
            device=maskeleyici_cihaz,
        )
        self.maske_isleyici = VaeImageProcessor(
            vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        self.pipeline = None
        if not sadece_maske:
            self.pipeline = CatVTONPipeline(
                base_ckpt="booksforcharlie/stable-diffusion-inpainting",
                attn_ckpt=repo,
                attn_ckpt_version="mix",
                weight_dtype=init_weight_dtype("bf16"),
                use_tf32=True,
                device="cuda",
            )

    def hazirla(self, kisi, kiyafet):
        """Görselleri modelin beklediği boyuta getirir (app.py ile aynı yöntem)."""
        kisi = resize_and_crop(kisi.convert("RGB"), self.boyut)
        kiyafet = resize_and_padding(kiyafet.convert("RGB"), self.boyut)
        return kisi, kiyafet

    def maskele(self, kisi, tur="upper"):
        """Döndürür: (orijinal AutoMasker maskesi, akıllı maske için ParseMaps)."""
        sonuc = self.maskeleyici(kisi, tur)
        haritalar = ParseMaps(*(_diziye_cevir(sonuc[k], kisi.size)
                                for k in ("densepose", "schp_lip", "schp_atr")))
        return sonuc["mask"].convert("L"), haritalar

    def yumusat(self, maske):
        """app.py'deki ile aynı kenar yumuşatma."""
        return self.maske_isleyici.blur(maske, blur_factor=9)

    def uret(self, kisi, kiyafet, maske, adim=20, cfg=2.5, seed=42):
        """Tek bir deneme üretir. Döndürür: (sonuç görseli, {'saniye', 'vram_gb'})"""
        if self.pipeline is None:
            raise RuntimeError("Motor sadece_maske=True ile açıldı, üretim yapılamaz.")
        uretec = torch.Generator(device="cuda").manual_seed(int(seed))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        sonuc = self.pipeline(
            image=kisi, condition_image=kiyafet, mask=maske,
            num_inference_steps=int(adim), guidance_scale=float(cfg),
            height=self.boyut[1], width=self.boyut[0], generator=uretec,
        )[0]
        bilgi = {"saniye": round(time.time() - t0, 1),
                 "vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)}
        torch.cuda.empty_cache()
        return sonuc, bilgi