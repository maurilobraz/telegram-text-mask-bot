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


def process_image(image_bytes: bytes) -> ExtractionResult:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return ExtractionResult(raw_text="", fields=[])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if max(h, w) < 2000:
        scale = 2000 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    pil = Image.fromarray(gray)
    text = pytesseract.image_to_string(pil, lang="por")

    logger.info(f"OCR TEXT:\n{text}")

    fields = []

    # SA - varios padroes
    sa_match = re.search(r'SA\s*[:;=]?\s*(\d{8})', text, re.IGNORECASE)
    if sa_match:
        fields.append(ExtractedField("SA", sa_match.group(1), 0.95))
    else:
        # Fallback: qualquer 8 digitos
        nums_8 = re.findall(r'\b(\d{8})\b', text)
        if nums_8:
            fields.append(ExtractedField("SA", nums_8[0], 0.7))

    # Telefone - "TELEFONE: XXXXXXXXXX"
    phone_match = re.search(r'(?:TELEFONE|TEL|CEL)\s*[:;]\s*(\d[\d\s\-\.]+)', text, re.IGNORECASE)
    if phone_match:
        fields.append(ExtractedField("TELEFONE", phone_match.group(1).strip(), 0.85))

    # Matricula - "MATRICULA: TT577337"
    mat_match = re.search(r'MATR[IÍ]CULA\s*[:;]\s*(\S+)', text, re.IGNORECASE)
    if mat_match:
        fields.append(ExtractedField("MATRICULA", mat_match.group(1).strip(), 0.9))

    # Percorre todas as linhas buscando "LABEL: VALOR"
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(.+?)\s*[:;]\s*(.+)$', line)
        if not m:
            continue

        label = m.group(1).strip().lower()
        value = m.group(2).strip()
        if not value:
            continue

        if "idcompanhia" in label or "id companhia" in label or "companhia" in label:
            if not any(f.label == "IDCOMPANHIA" for f in fields):
                fields.append(ExtractedField("IDCOMPANHIA", value, 0.8))

        elif "atividade" in label:
            if not any(f.label == "ATIVIDADE" for f in fields):
                fields.append(ExtractedField("ATIVIDADE", value, 0.8))

        elif "cliente" in label or "titular" in label:
            if not any(f.label == "CLIENTE" for f in fields):
                palavras = value.split()
                fields.append(ExtractedField("CLIENTE", " ".join(palavras[:2]), 0.8))

        elif "endereco" in label or "endereço" in label:
            if not any(f.label == "ENDERECO" for f in fields):
                fields.append(ExtractedField("ENDERECO", value, 0.8))

        elif "motivo" in label or "pendencia" in label:
            if not any(f.label == "MOTIVO" for f in fields):
                fields.append(ExtractedField("MOTIVO", value, 0.8))

    # Fallback: busca endereco comecando com rua/avenida/travessa/tv
    if not any(f.label == "ENDERECO" for f in fields):
        addr_match = re.search(r'(?:rua|avenida|av\.|tv\.|travessa)\s+[^\n]+', text, re.IGNORECASE)
        if addr_match:
            fields.append(ExtractedField("ENDERECO", addr_match.group(0).strip(), 0.6))

    # Fallback: busca nome do cliente
    if not any(f.label == "CLIENTE" for f in fields):
        name_match = re.search(r'(?:cliente|titular)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text, re.IGNORECASE)
        if name_match:
            nome = name_match.group(1).strip()
            palavras = nome.split()
            fields.append(ExtractedField("CLIENTE", " ".join(palavras[:2]), 0.7))

    return ExtractionResult(raw_text=text, fields=fields)
