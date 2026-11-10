"""Deney çıktılarını numaralı klasörlere kaydeder: 001_ad, 002_ad, ..."""
from pathlib import Path


def numarali_klasor(kok, ad):
    """kok altında sıradaki numarayla yeni klasör açar ve yolunu döndürür."""
    kok = Path(kok)
    kok.mkdir(parents=True, exist_ok=True)
    numaralar = [int(p.name[:3]) for p in kok.iterdir() if p.is_dir() and p.name[:3].isdigit()]
    klasor = kok / f"{max(numaralar, default=0) + 1:03d}_{ad}"
    klasor.mkdir()
    return klasor