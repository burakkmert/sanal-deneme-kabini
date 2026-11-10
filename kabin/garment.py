"""Kıyafet analizi: ürün fotoğrafından kıyafetin şeklini ölçer.

Fikir: kol boyunu, bolluğu ve boyu elle seçmek yerine kıyafetin siluetinden
çıkarmak. Bu dosya sadece ÖLÇER; ölçülerin maske ayarına çevrilmesi ayrı adım.
Düz arka planlı (beyaz/gri) ürün fotoğrafı varsayar. torch gerektirmez.
"""
import cv2
import numpy as np
from PIL import Image


def silhouette(image, max_side=640):
    """Kıyafetin siluetini (True/False dizisi) çıkarır; güvenilmezse None döndürür.

    Yöntem: GrabCut. Çerçeve "kesin arka plan", arka plan renginden farklı ya da
    kenar içeren bölge "olası kıyafet" olarak verilir; GrabCut sınırı kendisi çizer.
    İlk sürümdeki sabit eşik, açık zemindeki beyaz kıyafetlerde ve ekran
    alıntılarında bozuluyordu (deney 04: 17 fotoğrafın 5'i ölçülemedi).
    """
    im = image.convert("RGB")
    s = max_side / max(im.size)
    small = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s)))) if s < 1 else im
    rgb = np.array(small)
    h, w = rgb.shape[:2]
    k = max(3, min(h, w) // 40)

    # Arka plan rengi: çerçevedeki piksellerin ortancası (Lab renk uzayında)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
    frame = np.concatenate([lab[:k].reshape(-1, 3), lab[-k:].reshape(-1, 3),
                            lab[:, :k].reshape(-1, 3), lab[:, -k:].reshape(-1, 3)])
    dist = np.sqrt(((lab - np.median(frame, axis=0)) ** 2).sum(axis=2))
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    rough = ((dist > 9) | (cv2.Canny(gray, 15, 45) > 0)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, kernel)

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[rough > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(rough, kernel, iterations=2) > 0] = cv2.GC_FGD
    gc[:k] = gc[-k:] = cv2.GC_BGD
    gc[:, :k] = gc[:, -k:] = cv2.GC_BGD
    if (gc == cv2.GC_FGD).sum() < 50:
        return None
    try:
        cv2.grabCut(rgb, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    mask = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # En büyük parçanın dış hattını doldur (içteki baskılar, etiketler dolar)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    out = np.zeros((h, w), np.uint8)
    cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
    if not 0.06 < out.mean() < 0.9:
        return None
    return cv2.resize(out, im.size, interpolation=cv2.INTER_NEAREST) > 0


def _runs(row):
    """Bir satırdaki True parçalarının (başlangıç, bitiş) listesi."""
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def measure(image):
    """Kıyafetin ölçülerini döndürür (hepsi gövde genişliğine oranlı) ya da None.

    torso_w      : gövde genişliği (piksel)         body_len : gövde boyu (piksel)
    aspect       : boy / gövde genişliği
    sleeve_area  : tek kolun alanı / tüm silüet     (kolsuzda ~0)
    sleeve_reach : yakadan kol ucuna mesafe / yakadan etek köşesine mesafe
    width_ratio  : toplam genişlik / gövde genişliği
    """
    sil = silhouette(image)
    return None if sil is None else measure_silhouette(sil)


def measure_silhouette(sil):
    """measure() ile aynı, ama hazır bir silüet (True/False dizisi) alır."""
    ys, xs = np.nonzero(sil)
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    cx = (left + right) // 2
    height = bottom - top + 1

    # Her sütunun en alt pikseli. Gövde sütunlarında bu değer etek hizasındadır;
    # kol sütunlarında ya daha aşağıda (sarkan uzun kol) ya daha yukarıda (kısa kol).
    cols = np.arange(sil.shape[1])
    has = sil.any(axis=0)
    low = np.where(has, sil.shape[0] - 1 - np.argmax(sil[::-1], axis=0), -1)
    band = max(2, int(0.1 * (right - left)))
    hem = float(np.median(low[cx - band:cx + band + 1]))
    tol = 0.06 * height
    ok = has & (np.abs(low - hem) <= tol)
    if not ok[cx]:
        return None
    t_left = cx
    while t_left - 1 >= 0 and ok[t_left - 1]:
        t_left -= 1
    t_right = cx
    while t_right + 1 < len(cols) and ok[t_right + 1]:
        t_right += 1
    # Kol gövdeye bitişikse bir taraf fazla geniş çıkabilir: gövde simetrik kabul edilir.
    half = min(cx - t_left, t_right - cx)
    t_left, t_right = float(cx - half), float(cx + half)
    torso_w = t_right - t_left + 1
    body_top = int(np.argmax(sil[:, int(t_left):int(t_right) + 1].any(axis=1)))
    body_bottom = int(hem)
    body_len = body_bottom - body_top + 1

    # Kollar: gövde kenarından belirgin biçimde (gövde genişliğinin %12'si) dışarı
    # taşan silüet. Bu pay olmadan kolsuz kıyafetin yan kenarları "kol" sanılıyordu.
    # Kol uzunluğu ölçüsü: yaka noktasından kol ucuna mesafe / yaka noktasından
    # etek köşesine mesafe. Kol ucu etekten uzağa gidiyorsa uzun koldur.
    margin = 0.12 * torso_w
    neck = np.array([body_top, cx], dtype=np.float64)
    reaches, areas = [], []
    for side in ("left", "right"):
        region = sil.copy()
        if side == "left":
            region[:, max(0, int(t_left - margin)):] = False
            corner = np.array([body_bottom, t_left])
        else:
            region[:, :int(t_right + margin) + 1] = False
            corner = np.array([body_bottom, t_right])
        ry, rx = np.nonzero(region)
        areas.append(len(ry) / sil.sum())
        if len(ry) == 0:
            reaches.append(0.0)
            continue
        tip = np.sqrt((ry - neck[0]) ** 2 + (rx - neck[1]) ** 2).max()
        reaches.append(float(tip / np.linalg.norm(corner - neck)))
    sleeve_area = float(np.mean(areas))        # tek kolun silüete oranı
    sleeve_reach = float(np.mean(reaches))

    return {
        "torso_w": round(torso_w, 1),
        "body_len": int(body_len),
        "aspect": round(body_len / torso_w, 2),
        "sleeve_area": round(sleeve_area, 3),
        "sleeve_reach": round(sleeve_reach, 2),
        "width_ratio": round(float(right - left + 1) / torso_w, 2),
        "torso_box": [int(round(t_left)), int(body_top), int(round(t_right)), int(body_bottom)],
        "silhouette": sil,
    }


# Eşikler 4 örnek kıyafetten çıkarıldı. 17 fotoğraflık ilk sınav: 14 doğru.
# Bilinen sınırlar: (1) açık zeminde beyaz kıyafet ölçülemiyor, (2) kolları gövdeye
# yapışık asılı duran kıyafet kolsuz sanılıyor (dış hattan kol ayırt edilemiyor).
#   kolsuz yelek : alan 0.001           kısa kol (2 örnek): alan 0.04-0.09, erişim 0.62-0.81
#   uzun hırka   : alan 0.07, erişim 1.24
SLEEVE_MIN_AREA = 0.02
SLEEVE_LONG_REACH = 1.0


def classify_sleeve(m):
    """Ölçülerden kol boyunu tahmin eder: 'long' | 'short' | 'sleeveless'."""
    if m["sleeve_area"] < SLEEVE_MIN_AREA:
        return "sleeveless"
    return "long" if m["sleeve_reach"] >= SLEEVE_LONG_REACH else "short"


def debug_image(image, m):
    """Silüeti ve ölçülen gövde kutusunu fotoğrafın üstüne çizer."""
    img = np.array(image.convert("RGB")).astype(np.float32)
    sil = m["silhouette"]
    img[~sil] = img[~sil] * 0.35
    out = img.astype(np.uint8)
    x0, y0, x1, y1 = m["torso_box"]
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 40, 40), 3)
    contours, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (40, 220, 40), 2)
    return Image.fromarray(out)