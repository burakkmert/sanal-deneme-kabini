"""AutoMasker'ın ürettiği üç segmentasyon haritasının etiketleri.

  densepose : vücut bölgeleri (0-24)
  schp_lip  : LIP veri setiyle eğitilmiş ayrıştırıcı (0-19)
  schp_atr  : ATR veri setiyle eğitilmiş ayrıştırıcı (0-17)

Değerler CatVTON/model/cloth_masker.py ile aynıdır.
"""

# DensePose: kıyafetten bağımsız, vücudun kendisi
DENSEPOSE = {
    "torso": [1, 2],
    "hands": [3, 4],
    "feet": [5, 6],
    "thighs": [7, 8, 9, 10],        # uyluk
    "legs": [11, 12, 13, 14],       # diz altı
    "big_arms": [15, 16, 17, 18],   # üst kol (omuz-dirsek)
    "forearms": [19, 20, 21, 22],   # ön kol (dirsek-bilek)
    "face": [23, 24],               # baş (kulaklar dahil)
}

LIP = {
    "background": 0, "hat": 1, "hair": 2, "glove": 3, "sunglasses": 4,
    "upper_clothes": 5, "dress": 6, "coat": 7, "socks": 8, "pants": 9,
    "jumpsuits": 10, "scarf": 11, "skirt": 12, "face": 13,
    "left_arm": 14, "right_arm": 15, "left_leg": 16, "right_leg": 17,
    "left_shoe": 18, "right_shoe": 19,
}

ATR = {
    "background": 0, "hat": 1, "hair": 2, "sunglasses": 3,
    "upper_clothes": 4, "skirt": 5, "pants": 6, "dress": 7, "belt": 8,
    "left_shoe": 9, "right_shoe": 10, "face": 11, "left_leg": 12,
    "right_leg": 13, "left_arm": 14, "right_arm": 15, "bag": 16, "scarf": 17,
}

# Anlamsal grup -> (LIP etiket adları, ATR etiket adları)
# İki ayrıştırıcının BİRLEŞİMİNİ alıyoruz: biri kaçırırsa diğeri yakalar.
# (Orijinal kod yüz için sadece LIP'e bakıyor; kulak/boyun hatası oradan geliyor.)
PARSE_GROUPS = {
    "upper_clothes": (["upper_clothes"], ["upper_clothes"]),
    "coat": (["coat"], []),                      # ATR'de ayrı ceket sınıfı yok
    "dress": (["dress", "jumpsuits"], ["dress"]),
    "pants": (["pants"], ["pants"]),
    "skirt": (["skirt"], ["skirt"]),
    "face": (["face"], ["face"]),                # ATR yüzü boynu da içerir
    "hair": (["hair"], ["hair"]),
    "hat": (["hat"], ["hat"]),
    "sunglasses": (["sunglasses"], ["sunglasses"]),
    "glove": (["glove"], []),
    "scarf": (["scarf"], ["scarf"]),
    "bag": ([], ["bag"]),
    "belt": ([], ["belt"]),
    "shoes": (["left_shoe", "right_shoe"], ["left_shoe", "right_shoe"]),
    "socks": (["socks"], []),
    "arms_skin": (["left_arm", "right_arm"], ["left_arm", "right_arm"]),
    "legs_skin": (["left_leg", "right_leg"], ["left_leg", "right_leg"]),
}