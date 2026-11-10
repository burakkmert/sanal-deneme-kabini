"""Kıyafet analizi, ikinci yöntem: CLIP ile sıfır-örnek (zero-shot) sınıflandırma.

CLIP, bir görselin bir cümleye ne kadar uyduğunu ölçen hazır eğitilmiş bir model.
Kıyafet fotoğrafını "uzun kollu üst", "kısa kollu üst", "kolsuz üst" cümleleriyle
karşılaştırıp en çok uyanı seçiyoruz. Eğitim yok, sadece soru soruyoruz.
Kural tabanlı yöntemle (garment.py) aynı fotoğraflarda karşılaştırmak için yazıldı.
"""
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Her özellik için: etiket -> o etiketi anlatan cümleler (ortalaması alınır)
SORULAR = {
    "sleeve": {
        "long": ["a product photo of a long sleeve top", "a shirt with long sleeves down to the wrists"],
        "short": ["a product photo of a short sleeve top", "a t-shirt with short sleeves"],
        "sleeveless": ["a product photo of a sleeveless top", "a tank top with no sleeves"],
    },
    "fit": {
        "tight": ["a tight slim fit top", "a fitted body-hugging top"],
        "normal": ["a regular fit top", "a top with a standard fit"],
        "loose": ["an oversized loose top", "a baggy boxy oversized top"],
    },
    "length": {
        "crop": ["a cropped top that ends above the waist", "a short crop top"],
        "normal": ["a regular length top that ends at the hips", "a standard length top"],
        "long": ["a long top that covers the hips", "a longline tunic top"],
    },
}


def kareye_tamamla(image):
    """Görseli kenar rengiyle kareye tamamlar. CLIP girişi kare; tamamlamazsak
    uzun görsellerin altı (kol uçları, etek) kırpılır."""
    im = image.convert("RGB")
    w, h = im.size
    kenar = np.array(im)[[0, -1], :, :].reshape(-1, 3)
    renk = tuple(int(v) for v in np.median(kenar, axis=0))
    yan = max(w, h)
    tuval = Image.new("RGB", (yan, yan), renk)
    tuval.paste(im, ((yan - w) // 2, (yan - h) // 2))
    return tuval


class KiyafetCLIP:
    def __init__(self, model="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model)
        # Cümlelerin vektörleri bir kez hesaplanır
        self.metin = {}
        with torch.no_grad():
            for ozellik, etiketler in SORULAR.items():
                vektorler = []
                for cumleler in etiketler.values():
                    girdi = self.processor(text=cumleler, return_tensors="pt", padding=True).to(self.device)
                    v = self.model.get_text_features(**girdi)
                    v = v / v.norm(dim=-1, keepdim=True)
                    v = v.mean(dim=0)
                    vektorler.append(v / v.norm())
                self.metin[ozellik] = torch.stack(vektorler)

    @torch.no_grad()
    def classify(self, image):
        """Döndürür: {'sleeve': 'long', 'sleeve_p': {'long': 0.91, ...}, 'fit': ..., 'length': ...}"""
        girdi = self.processor(images=kareye_tamamla(image), return_tensors="pt").to(self.device)
        g = self.model.get_image_features(**girdi)
        g = g / g.norm(dim=-1, keepdim=True)
        sonuc = {}
        for ozellik, etiketler in SORULAR.items():
            olasilik = (100.0 * g @ self.metin[ozellik].T).softmax(dim=-1)[0].cpu().numpy()
            adlar = list(etiketler)
            sonuc[ozellik] = adlar[int(olasilik.argmax())]
            sonuc[ozellik + "_p"] = {a: round(float(p), 2) for a, p in zip(adlar, olasilik)}
        return sonuc


# CLIP'in boy tahmini -> etek payı (piksel, 1024 yüksekliğe göre).
# Deney 003 (hırka, 40) ve 006 (bol forma, 70) sonuçlarından; örnek arttıkça ayarlanacak.
ETEK_PX = {"crop": 0, "normal": 35, "long": 70}


def maske_ayarlari(c, en_az=0.6):
    """CLIP çıktısını build_upper_mask ayarlarına çevirir.

    'emin' False ise kol tahmininin olasılığı en_az'ın altındadır; arayüzde
    kullanıcıya sorulmalıdır.
    """
    return {
        "sleeve": c["sleeve"],
        "fit": c["fit"],
        "hem_px": ETEK_PX[c["length"]],
        "emin": c["sleeve_p"][c["sleeve"]] >= en_az,
    }