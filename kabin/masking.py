"""Kıyafete duyarlı maske (v1: sadece üst giyim + kol boyu).

    maske = (eski kıyafetin alanı  ∪  yeni kıyafetin kaplayacağı alan) − korunan bölgeler

Orijinal AutoMasker "korunmayan her şeyi" maskeleyip üstüne dışbükey zarf çekiyor.
Biz tersini yapıyoruz: sadece gereken bölgeleri dahil ediyoruz.
Bu dosya torch gerektirmez (numpy + OpenCV), GPU olmadan test edilebilir.
"""
import cv2
import numpy as np

from .labels import ATR, DENSEPOSE, LIP, PARSE_GROUPS

SLEEVES = ("long", "short", "sleeveless")


class ParseMaps:
    """Üç haritadan anlamsal bölgeleri (True/False dizisi olarak) çıkarır."""

    def __init__(self, densepose, lip, atr):
        if not (densepose.shape == lip.shape == atr.shape):
            raise ValueError("Harita boyutları aynı olmalı")
        self.dp, self.lip, self.atr = densepose, lip, atr
        self.shape = densepose.shape

    def body(self, part):
        """DensePose vücut bölgesi, örn. body('forearms')."""
        return np.isin(self.dp, DENSEPOSE[part])

    def parse(self, group):
        """LIP ve ATR'nin birleşimi, örn. parse('upper_clothes')."""
        lip_names, atr_names = PARSE_GROUPS[group]
        return (np.isin(self.lip, [LIP[n] for n in lip_names])
                | np.isin(self.atr, [ATR[n] for n in atr_names]))


def _kernel(px):
    size = 2 * int(px) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def dilate(mask, px):
    """Maskeyi her yöne px piksel genişletir."""
    if px <= 0:
        return mask.copy()
    return cv2.dilate(mask.astype(np.uint8), _kernel(px)) > 0


def close(mask, px):
    """Küçük boşlukları ve çatlakları kapatır."""
    if px <= 0:
        return mask.copy()
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(px)) > 0


def fill_holes(mask):
    """Maskenin içinde kalan kapalı delikleri doldurur."""
    padded = np.pad(mask.astype(np.uint8), 1)
    ff = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, ff, (0, 0), 1)       # dışarıdan ulaşılabilen her yeri boya
    return mask | (padded[1:-1, 1:-1] == 0)     # boyanmayan siyahlar = delik


def remove_small(mask, min_ratio=0.001):
    """Görüntünün binde birinden küçük kopuk parçaları siler (kulaktaki nokta gibi)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_ratio * mask.size
    return keep[labels]


def erode(mask, px):
    """Maskeyi her yönden px piksel daraltır."""
    if px <= 0:
        return mask.copy()
    return cv2.erode(mask.astype(np.uint8), _kernel(px)) > 0


def open_thin(mask, px):
    """px'ten ince şeritleri ve çıkıntıları siler, ana şekli korur."""
    if px <= 0:
        return mask.copy()
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _kernel(px)) > 0


def protect_area(maps, old, scale=1.0, hem_px=6):
    """Hiçbir koşulda maskeye girmeyecek bölgeler.

    old: eski kıyafetin alanı. Bir piksele bir ayrıştırıcı 'üst giysi', diğeri
    'pantolon' derse üst giysi kazanır (yoksa etek ucu maskesiz kalır).
    """
    b, p = maps.body, maps.parse
    # Baş çekirdeği: DensePose başı + LIP yüzü (ikisi de dar ve güvenilir).
    core = b("face") | (maps.lip == LIP["face"])
    # ATR 'yüz' etiketi omuzlara ve yakaya taşabiliyor. Onu sadece başın
    # hemen yanında (kulak gibi), gövdenin ve eski kıyafetin dışında sayıyoruz.
    atr_face = maps.atr == ATR["face"]
    head = core | (atr_face & dilate(core, round(15 * scale)) & ~b("torso") & ~old)
    # Alt giyim: eski üstle çakışan pikseller hariç, kenardan hem_px içeri çekilmiş.
    # Böylece yeni kıyafetin eteği bel çizgisinin biraz üstüne binebilir.
    lower = erode((p("pants") | p("skirt")) & ~old, round(hem_px * scale))
    return (head | p("hair") | p("hat") | p("sunglasses")
            | b("hands") | p("glove") | b("feet") | p("shoes") | p("bag") | lower)


def build_upper_mask(maps, sleeve="long", grow_px=12):
    """Üst giyim maskesi. Döndürür: {'mask': uint8 0/255, 'old', 'new', 'protect'}"""
    if sleeve not in SLEEVES:
        raise ValueError(f"sleeve {SLEEVES} içinden olmalı")
    scale = maps.shape[0] / 1024.0              # piksel ayarları 1024 yüksekliğe göre
    b, p = maps.body, maps.parse

    # 1) Eski kıyafetin kapladığı alan: silinmesi gereken kumaş
    old = p("upper_clothes") | p("coat")

    # 2) Yeni kıyafetin kaplayacağı alan: gövde + kol boyuna göre kollar
    new = b("torso")
    if sleeve in ("long", "short"):
        new = new | b("big_arms")
    if sleeve == "long":
        new = new | b("forearms")

    # 3) Korunan bölgeler
    protect = protect_area(maps, old, scale)

    # 4) Birleştir, temizle, kenar payı ekle, korunanları çıkar
    mask = old | new
    mask = close(mask, round(6 * scale))
    mask = fill_holes(mask)
    mask = dilate(mask, round(grow_px * scale))
    mask = mask & ~protect
    mask = open_thin(mask, round(5 * scale))    # ince şerit ve kuyrukları temizle
    mask = remove_small(mask)

    return {"mask": mask.astype(np.uint8) * 255, "old": old, "new": new, "protect": protect}