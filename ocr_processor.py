import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
from dataclasses import dataclass, field


@dataclass
class ExtractedField:
    label: str
    value: str
    confidence: float = 0.0


@dataclass
class ExtractionResult:
    raw_text: str
    fields: list[ExtractedField] = field(default_factory=list)


FIELD_MAP = {
    "sa": "SA",
    "numero do sa": "SA",
    "n do sa": "SA",
    "nº sa": "SA",
    "num sa": "SA",
    "n° sa": "SA",
    "idcompanhia": "IDCOMPANHIA",
    "id companhia": "IDCOMPANHIA",
    "companhia": "IDCOMPANHIA",
    "atividade": "ATIVIDADE",
    "tipo de atividade": "ATIVIDADE",
    "tipo atividade": "ATIVIDADE",
    "cliente": "CLIENTE",
    "nome do cliente": "CLIENTE",
    "titular": "CLIENTE",
    "endereco": "ENDERECO",
    "endereço": "ENDERECO",
    "telefone": "TELEFONE",
    "celular": "TELEFONE",
    "matricula": "MATRICULA",
    "nome do tecnico": "NOME_TECNICO",
    "tecnico": "NOME_TECNICO",
    "motivo": "MOTIVO",
    "motivo da pendencia": "MOTIVO",
    "nome recebeu": "NOME_RECEBEU",
    "recebeu": "NOME_RECEBEU",
    "contato": "CONTATO",
}


def load_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Nao foi possivel carregar a imagem")
    return img


def upscale(img: np.ndarray, factor: int = 3) -> np.ndarray:
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def ocr_single(img: np.ndarray, lang: str = "por", config: str = "") -> str:
    pil = Image.fromarray(img)
    return pytesseract.image_to_string(pil, lang=lang, config=config).strip()


def ocr_full_image(image_bytes: bytes) -> str:
    """Roda OCR na imagem inteira com varias configs e retorna o melhor texto."""
    img = load_image(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    versions = []

    # OTSU padrao
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(otsu)
    versions.append(cv2.bitwise_not(otsu))

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    versions.append(clahe.apply(gray))

    # Adaptive
    versions.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2))

    # Denoise
    versions.append(cv2.fastNlMeansDenoising(gray, None, 10, 7, 21))

    configs = [
        r"--oem 3 --psm 6",
        r"--oem 3 --psm 4",
        r"--oem 3 --psm 3",
    ]

    all_texts = []
    for v in versions:
        big = upscale(v, 3)
        for cfg in configs:
            try:
                t = ocr_single(big, lang, cfg)
                if t:
                    all_texts.append(t)
            except Exception:
                continue

    # Tambem tenta na imagem original colorida
    big_color = upscale(img, 3)
    for cfg in configs[:2]:
        try:
            t = ocr_single(big_color, lang, cfg)
            if t:
                all_texts.append(t)
        except Exception:
            continue

    return "\n\n".join(all_texts)


def extract_sa_from_yellow(image_bytes: bytes) -> str:
    """Busca SA na regiao amarela no topo da imagem."""
    img = load_image(image_bytes)
    h, w = img.shape[:2]

    top = img[0:int(h * 0.30), :]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([40, 255, 255]))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    yellow_only = cv2.bitwise_and(top, top, mask=mask)
    gray = cv2.cvtColor(yellow_only, cv2.COLOR_BGR2GRAY)

    for thresh_val in [0, 80, 100, 120]:
        if thresh_val == 0:
            _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, t = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        big = upscale(t, 4)
        big = cv2.medianBlur(big, 3)

        for cfg in [r"--oem 3 --psm 7", r"--oem 3 --psm 8"]:
            try:
                text = ocr_single(big, "por", cfg)
                nums = re.findall(r'\d{8}', text)
                if nums:
                    return nums[0]
            except Exception:
                continue

    return ""


def normalize_label(raw: str) -> str:
    """Normaliza o label encontrado no OCR."""
    clean = raw.lower().strip()
    clean = re.sub(r'[^a-záéíóúãõêô\s]', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return FIELD_MAP.get(clean, clean.upper().replace(" ", "_"))


def parse_label_value_lines(text: str) -> list[tuple[str, str]]:
    """
    Le o texto linha por linha e tenta separar label (esquerda) e valor (direita).
    O print tem: label ...... valor
    """
    results = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 3:
            continue

        # Padrao 1: "label: valor"
        m = re.match(r'^([^:]+?)\s*[:;]\s*(.+)$', line)
        if m:
            results.append((m.group(1).strip(), m.group(2).strip()))
            continue

        # Padrao 2: "label   valor" (espacos no meio)
        m = re.match(r'^([a-zA-Záéíóúãõêô\s]{2,30})\s{2,}(.+)$', line)
        if m:
            results.append((m.group(1).strip(), m.group(2).strip()))
            continue

    return results


def extract_fields_from_text(text: str) -> list[ExtractedField]:
    """Extrai campos do texto OCR usando o mapa de campos do print."""
    fields = []
    seen = set()

    # SA - 8 digitos no topo
    sa_match = re.search(r'\b(\d{8})\b', text)
    if sa_match:
        fields.append(ExtractedField("SA", sa_match.group(1), 0.9))
        seen.add("SA")

    # Telefone
    phone_match = re.search(r'\(?\d{2}\)?\s*\d{4,5}[-.\s]?\d{4}', text)
    if phone_match:
        fields.append(ExtractedField("TELEFONE", phone_match.group(0).strip(), 0.85))
        seen.add("TELEFONE")

    # Parse label: valor
    pairs = parse_label_value_lines(text)
    for raw_label, value in pairs:
        norm = normalize_label(raw_label)
        if norm in seen or len(value) < 1:
            continue
        seen.add(norm)

        # Cliente: max 2 nomes se for grande
        if norm == "CLIENTE":
            palavras = value.split()
            value = " ".join(palavras[:2])

        fields.append(ExtractedField(norm, value, 0.8))

    return fields


def process_image(image_bytes: bytes) -> ExtractionResult:
    """Processa uma imagem e retorna os campos extraidos."""
    # 1. OCR na imagem inteira
    raw_text = ocr_full_image(image_bytes)

    # 2. SA da regiao amarela (mais confiavel)
    sa_yellow = extract_sa_from_yellow(image_bytes)

    # 3. Extrai campos do texto
    fields = extract_fields_from_text(raw_text)

    # 4. Sobrescreve SA se achou na regiao amarela
    if sa_yellow:
        found = False
        for f in fields:
            if f.label == "SA":
                f.value = sa_yellow
                f.confidence = 0.95
                found = True
                break
        if not found:
            fields.insert(0, ExtractedField("SA", sa_yellow, 0.95))

    return ExtractionResult(raw_text=raw_text, fields=fields)
