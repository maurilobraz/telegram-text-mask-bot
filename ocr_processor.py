import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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


def ocr_simple(image_bytes: bytes) -> str:
    """OCR simples e direto."""
    img = load_image(image_bytes)
    
    # Aumenta a imagem
    h, w = img.shape[:2]
    if max(h, w) < 2000:
        scale = 2000 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Converte para cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tenta direto
    pil = Image.fromarray(gray)
    text = pytesseract.image_to_string(pil, lang="por")
    
    logger.info(f"OCR simples: {text[:500]}")
    return text.strip()


def normalize_label(raw: str) -> str:
    clean = raw.lower().strip()
    clean = re.sub(r'[^a-záéíóúãõêô\s]', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return FIELD_MAP.get(clean, clean.upper().replace(" ", "_"))


def parse_label_value_lines(text: str) -> list[tuple[str, str]]:
    results = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 3:
            continue

        # "label: valor"
        m = re.match(r'^([^:]+?)\s*[:;]\s*(.+)$', line)
        if m:
            results.append((m.group(1).strip(), m.group(2).strip()))
            continue

        # "label   valor" (espacos no meio)
        m = re.match(r'^([a-zA-Záéíóúãõêô\s]{2,30})\s{2,}(.+)$', line)
        if m:
            results.append((m.group(1).strip(), m.group(2).strip()))
            continue

    return results


def extract_fields_from_text(text: str) -> list[ExtractedField]:
    fields = []
    seen = set()

    # SA - 8 digitos
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

        if norm == "CLIENTE":
            palavras = value.split()
            value = " ".join(palavras[:2])

        fields.append(ExtractedField(norm, value, 0.8))

    return fields


def process_image(image_bytes: bytes) -> ExtractionResult:
    raw_text = ocr_simple(image_bytes)
    fields = extract_fields_from_text(raw_text)
    
    logger.info(f"Campos: {[(f.label, f.value) for f in fields]}")
    
    return ExtractionResult(raw_text=raw_text, fields=fields)
