"""Kıyafete duyarlı maske (v3: üst giyim; kol boyu + bolluk + etek payı + yaka payı).

    maske = (eski kıyafetin alanı  ∪  yeni kıyafetin kaplayacağı alan) − korunan bölgeler

Orijinal AutoMasker "korunmayan her şeyi" maskeleyip üstüne dışbükey zarf çekiyor.
Biz tersini yapıyoruz: sadece gereken bölgeleri dahil ediyoruz.
Bu dosya torch gerektirmez (numpy + OpenCV), GPU olmadan test edilebilir.
"""
import cv2
import numpy as np

from .labels import ATR, DENSEPOSE, LIP, PARSE_GROUPS

SLEEVES = ("long", "short", "sleeveless")
# Bolluk -> kenar payı (piksel, 1024 yüksekliğe göre). Deney 001/002: 12 piksel
# payla bol hırkanın kolu dirsekte kaldı, 35 pikselle bileğe indi. Model maskenin
# şeklini kıyafetin silueti için ipucu olarak kullanıyor; bol kıyafete yer lazım.
FIT_GROW = {"tight": 12, "normal": 24, "loose": 36}


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


def extend_down(mask, px):
    """Maskeyi sadece AŞAĞI doğru px piksel uzatır (etek payı için)."""
    if px <= 0:
        return mask.copy()
    kernel = np.ones((int(px) + 1, 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, anchor=(0, int(px))) > 0


def open_thin(mask, px):
    """px'ten ince şeritleri ve çıkıntıları siler, ana şekli korur."""
    if px <= 0:
        return mask.copy()
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _kernel(px)) > 0


def split_head(maps, neck=0.6):
    """Başı 'korunacak kısım' ve 'serbest boyun' olarak ikiye ayırır.

    DensePose 'baş' bölgesi boynu da içerir. Boynun tamamı korunursa yakalı
    kıyafetin yakası sıkışıp bozuluyor (deney 005). Bu yüzden boynun alt kısmı
    maskeye açılır:  neck=0 -> boyun tamamen korunur, neck=1 -> çeneye kadar açılır.
    Döndürür: (korunan_bas, serbest_boyun)
    """
    dp_head = maps.body("face")
    lip_face = maps.lip == LIP["face"]
    free = np.zeros(maps.shape, dtype=bool)
    if neck > 0 and lip_face.any() and dp_head.any():
        chin = np.nonzero(lip_face.any(axis=1))[0].max()       # LIP yüzü çenede biter
        bottom = np.nonzero(dp_head.any(axis=1))[0].max()      # boynun en alt satırı
        if bottom > chin:
            cut = int(round(bottom - neck * (bottom - chin)))
            below = np.arange(maps.shape[0])[:, None] > cut
            free = dp_head & below & ~lip_face
    return (dp_head & ~free) | lip_face, free


def protect_area(maps, old, scale=1.0, overlap_px=6, hem_zone=None, neck=0.6):
    """Hiçbir koşulda maskeye girmeyecek bölgeler.

    old: eski kıyafetin alanı. Bir piksele bir ayrıştırıcı 'üst giysi', diğeri
    'pantolon' derse üst giysi kazanır (yoksa etek ucu maskesiz kalır).
    hem_zone: yeni kıyafetin eteğinin sarkacağı alan; pantolon koruması buradan kalkar.
    """
    b, p = maps.body, maps.parse
    # Baş çekirdeği: DensePose başı (serbest boyun hariç) + LIP yüzü.
    core, free_neck = split_head(maps, neck)
    # Boğazlı kazak gibi boynu örten eski kıyafet korunmaz, yoksa çenenin altında
    # eski kumaştan bir şerit kalır. LIP'in "yüz" dediği yer her durumda korunur.
    core = (core & ~old) | (maps.lip == LIP["face"])
    # ATR 'yüz' etiketi omuzlara ve yakaya taşabiliyor. Onu sadece başın hemen
    # yanında (kulak gibi), gövdenin, eski kıyafetin ve serbest boynun dışında sayıyoruz.
    atr_face = maps.atr == ATR["face"]
    near = dilate(core, round(15 * scale)) & ~dilate(free_neck, round(15 * scale))
    head = core | (atr_face & near & ~b("torso") & ~old)
    # Alt giyim: eski üstle çakışan pikseller hariç, kenardan biraz içeri çekilmiş.
    # Böylece yeni kıyafetin eteği bel çizgisinin biraz üstüne binebilir.
    lower = erode((p("pants") | p("skirt")) & ~old, round(overlap_px * scale))
    if hem_zone is not None:
        lower = lower & ~hem_zone
    # Eller: DensePose "el" bölgesi bileği ve manşeti de kapsayabiliyor. Eski kıyafetin
    # kumaşı el korumasına girerse yeni kıyafetin altında kalıyor (deney 008: siyah
    # manşetler, deney 010: pembe gömlek parçaları). Kumaş olan piksel el sayılmaz.
    hands = b("hands") & ~old
    return (head | p("hair") | p("hat") | p("sunglasses")
            | hands | p("glove") | b("feet") | p("shoes") | p("bag") | lower)


def build_upper_mask(maps, sleeve="long", fit="normal", hem_px=0, neck=0.6, grow_px=None):
    """Üst giyim maskesi.

    sleeve : yeni kıyafetin kolu      -> long | short | sleeveless
    fit    : yeni kıyafetin bolluğu   -> tight | normal | loose  (kenar payını belirler)
    hem_px : etek payı; yeni kıyafet eskisinden ne kadar aşağı sarkabilir (piksel)
    neck   : boynun ne kadarı yakaya açılsın (0 = hiç, 1 = çeneye kadar)
    grow_px: verilirse fit'in kenar payını ezer (deney için)
    Döndürür: {'mask': uint8 0/255, 'old', 'new', 'protect'}
    """
    if sleeve not in SLEEVES:
        raise ValueError(f"sleeve {SLEEVES} içinden olmalı")
    if fit not in FIT_GROW:
        raise ValueError(f"fit {tuple(FIT_GROW)} içinden olmalı")
    if grow_px is None:
        grow_px = FIT_GROW[fit]
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

    # 2b) Etek payı: eski kıyafetin gövde kısmını aşağı doğru uzat.
    #     Kollar hariç tutulur, yoksa uzantı ellerin yanına iner.
    hem_zone = None
    if hem_px > 0:
        arms = b("big_arms") | b("forearms") | b("hands")
        lower_clothes = p("pants") | p("skirt")
        body_part = (old & ~arms) if old.any() else (b("torso") & ~lower_clothes)
        hem_zone = extend_down(body_part, round(hem_px * scale))
        new = new | hem_zone

    # 2c) Yaka payı: boynun alt kısmı yeni kıyafetin yakasına açılır
    new = new | split_head(maps, neck)[1]

    # 3) Korunan bölgeler
    protect = protect_area(maps, old, scale, hem_zone=hem_zone, neck=neck)
    # 3b) Yeni kıyafetin örtmeyeceği çıplak kol korunur. Sadece "dahil etmemek"
    #     yetmiyor: kenar payı gövdeden kolun iç yüzüne taşıp teni bozuyor (deney 006).
    #     Eski kıyafetin kumaşı varsa (uzun koldan kısaya geçiş) orası yine silinir.
    if sleeve == "short":
        protect = protect | (b("forearms") & ~old)
    elif sleeve == "sleeveless":
        protect = protect | ((b("big_arms") | b("forearms")) & ~old)

    # 4) Birleştir, temizle, kenar payı ekle, korunanları çıkar
    mask = old | new
    mask = close(mask, round(6 * scale))
    mask = fill_holes(mask)
    mask = dilate(mask, round(grow_px * scale))
    mask = mask & ~protect
    mask = open_thin(mask, round(5 * scale))    # ince şerit ve kuyrukları temizle
    mask = remove_small(mask)

    return {"mask": mask.astype(np.uint8) * 255, "old": old, "new": new, "protect": protect}


# ---------------------------------------------------------------------------
# Melez maske: orijinal AutoMasker maskesi + kıyafete göre düzeltmeler
# ---------------------------------------------------------------------------
# build_upper_mask (yukarıda) maskeyi sıfırdan kuruyordu. Kolay fotoğrafta iyi
# çalıştı ama segmentasyon hata yaptığında (boğazlı kazak, manşet, el çevresi)
# eski kıyafetin parçaları yerinde kaldı (deney 008-010). Orijinal maske "emin
# olmadığın her yeri maskele" dediği için bu hatalara dayanıklı. Bu yüzden onu
# temel alıp yalnızca deneyde işe yaradığını gördüğümüz düzeltmeleri ekliyoruz.

def bare_ratio(maps, part, old):
    """Vücut bölgesinin ne kadarı çıplak ten (0-1). Bölge görünmüyorsa 0."""
    region = maps.body(part)
    if not region.any():
        return 0.0
    bare = region & maps.parse("arms_skin") & ~old
    return float(bare.sum() / region.sum())


def build_hybrid_mask(baseline, maps, sleeve="long", hem_px=0, bare_min=0.8):
    """Orijinal maskeyi kıyafete göre düzeltir.

    baseline: AutoMasker'ın 'upper' maskesi (PIL ya da dizi)
    Düzeltmeler:
      1) Etek payı: yeni kıyafet eskisinden uzunsa maske aşağı uzatılır.
      2) Çıplak kol koruması: yeni kıyafet kolu örtmeyecekse VE kol zaten
         çıplaksa (bare_min oranında) kişinin kendi kolu korunur. Kol kumaşla
         kaplıysa dokunulmaz; orijinal maske gibi tamamı modele bırakılır.
      3) Kopuk küçük lekeler (kulaktaki nokta gibi) silinir.
    Döndürür: {'mask': uint8 0/255, 'notes': [uygulanan düzeltmeler]}
    """
    if sleeve not in SLEEVES:
        raise ValueError(f"sleeve {SLEEVES} içinden olmalı")
    mask = np.array(baseline.convert("L") if hasattr(baseline, "convert") else baseline) > 127
    if mask.shape != maps.shape:
        raise ValueError("Maske ve harita boyutları farklı")
    scale = maps.shape[0] / 1024.0
    b, p = maps.body, maps.parse
    old = p("upper_clothes") | p("coat")
    hands = b("hands") & ~old
    notes = []

    # 1) Etek payı
    keep = old | b("torso")
    if hem_px > 0:
        arms = b("big_arms") | b("forearms") | b("hands")
        lower = p("pants") | p("skirt")
        body_part = (old & ~arms) if (old & ~arms).any() else (b("torso") & ~lower)
        # Kadrajda alt giyimden az yer kaldıysa etek payı kısılır. Maske kadrajın
        # dibine kadar inerse model pantolonun rengini göremez ve uydurur (deney 2:
        # bej pantolon siyaha döndü). Görünen alt giyimin en fazla üçte biri kapatılır.
        hem = round(hem_px * scale)
        if body_part.any() and lower.any():
            room = int(np.nonzero(lower.any(axis=1))[0].max()) - int(np.nonzero(body_part.any(axis=1))[0].max())
            hem = max(0, min(hem, room // 3))
        hem_zone = extend_down(body_part, hem) & ~hands & ~p("shoes")
        mask = mask | hem_zone
        keep = keep | hem_zone
        notes.append(f"etek payı {hem}px" + ("" if hem == round(hem_px * scale) else f" (istenen {hem_px}, kadraj dar)"))

    # 2) Çıplak kol koruması (sadece güvenliyken)
    parts = {"long": [], "short": ["forearms"], "sleeveless": ["big_arms", "forearms"]}[sleeve]
    if parts and all(bare_ratio(maps, part, old) >= bare_min for part in parts):
        arm = np.zeros(maps.shape, dtype=bool)
        for part in parts:
            arm |= b(part)
        arm = (arm | hands) & ~old
        if sleeve == "short":
            keep = keep | b("big_arms")
        # Kolun çevresindeki maske halkasını da kaldır, ama gövde tarafına dokunma
        remove = dilate(arm, round(30 * scale)) & ~dilate(keep, round(20 * scale))
        mask = mask & ~(remove | arm)
        mask = open_thin(mask, round(8 * scale))
        notes.append("çıplak kol korundu: " + ", ".join(parts))
    elif parts:
        notes.append("kol kumaşla kaplı: orijinal maske korundu")

    # 3) Kopuk lekeler
    mask = remove_small(mask, 0.002)
    return {"mask": mask.astype(np.uint8) * 255, "notes": notes}