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

    sa_match = re.search(r'SA[\s\-:]?\s*(\d{8})', text, re.IGNORECASE)
    if sa_match:
        fields.append(ExtractedField("SA", sa_match.group(1), 0.95))

    phone_match = re.search(r'\(?\d{2}\)?\s*\d{4,5}[-.\s]?\d{4}', text)
    if phone_match:
        fields.append(ExtractedField("TELEFONE", phone_match.group(0).strip(), 0.85))

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(.+?)\s*[:;]\s*(.+)$', line)
        if m:
            label = m.group(1).strip().lower()
            value = m.group(2).strip()
            if len(value) < 1:
                continue

            if "atividade" in label:
                fields.append(ExtractedField("ATIVIDADE", value, 0.8))
            elif "cliente" in label or "titular" in label:
                palavras = value.split()
                fields.append(ExtractedField("CLIENTE", " ".join(palavras[:2]), 0.8))
            elif "endereco" in label or "endereço" in label:
                fields.append(ExtractedField("ENDERECO", value, 0.8))
            elif "idcompanhia" in label or "id companhia" in label or "companhia" in label:
                fields.append(ExtractedField("IDCOMPANHIA", value, 0.8))
            elif "matricula" in label:
                fields.append(ExtractedField("MATRICULA", value, 0.8))
            elif "motivo" in label:
                fields.append(ExtractedField("MOTIVO", value, 0.8))
            elif "telefone" in label or "celular" in label:
                if not any(f.label == "TELEFONE" for f in fields):
                    fields.append(ExtractedField("TELEFONE", value, 0.8))

    return ExtractionResult(raw_text=text, fields=fields)
