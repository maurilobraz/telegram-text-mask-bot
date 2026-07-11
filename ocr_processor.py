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

        # Com dois pontos: "label: valor"
        m = re.match(r'^(.+?)\s*[:;]\s*(.+)$', line)
        if m:
            label = m.group(1).strip().lower()
            value = m.group(2).strip()
            if len(value) < 1:
                continue
            _classify_field(fields, label, value)
            continue

        # Sem dois pontos: "label   valor" (espacos no meio)
        m = re.match(r'^([a-zA-Záéíóúãõêô\s]{2,30})\s{2,}(.+)$', line)
        if m:
            label = m.group(1).strip().lower()
            value = m.group(2).strip()
            if len(value) < 1:
                continue
            _classify_field(fields, label, value)
            continue

    # Se nao achou cliente, tenta buscar por "nome" seguido de palavras
    if not any(f.label == "CLIENTE" for f in fields):
        name_match = re.search(r'(?:nome|cliente|titular)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text, re.IGNORECASE)
        if name_match:
            nome = name_match.group(1).strip()
            palavras = nome.split()
            fields.append(ExtractedField("CLIENTE", " ".join(palavras[:2]), 0.7))

    # Se nao achou endereco, tenta buscar padrao de endereco no texto
    if not any(f.label == "ENDERECO" for f in fields):
        addr_match = re.search(r'(?:rua|av\.|avenida|alameda|travessa|beco|praca|estrada|rodovia|bairro)\s+[^\n]+', text, re.IGNORECASE)
        if addr_match:
            fields.append(ExtractedField("ENDERECO", addr_match.group(0).strip(), 0.6))

    return ExtractionResult(raw_text=text, fields=fields)


def _classify_field(fields: list, label: str, value: str):
    """Classifica um campo label:valor e adiciona a lista."""
    labels_ja_tem = {f.label for f in fields}

    if "atividade" in label:
        if "ATIVIDADE" not in labels_ja_tem:
            fields.append(ExtractedField("ATIVIDADE", value, 0.8))

    elif "cliente" in label or "titular" in label or "nome do" in label:
        palavras = value.split()
        nome = " ".join(palavras[:2])
        if "CLIENTE" not in labels_ja_tem:
            fields.append(ExtractedField("CLIENTE", nome, 0.8))

    elif "endereco" in label or "endereço" in label or "end" in label or "rua" in label or "avenida" in label:
        if "ENDERECO" not in labels_ja_tem:
            fields.append(ExtractedField("ENDERECO", value, 0.8))

    elif "idcompanhia" in label or "id companhia" in label or "companhia" in label:
        if "IDCOMPANHIA" not in labels_ja_tem:
            fields.append(ExtractedField("IDCOMPANHIA", value, 0.8))

    elif "matricula" in label:
        if "MATRICULA" not in labels_ja_tem:
            fields.append(ExtractedField("MATRICULA", value, 0.8))

    elif "motivo" in label:
        if "MOTIVO" not in labels_ja_tem:
            fields.append(ExtractedField("MOTIVO", value, 0.8))

    elif "telefone" in label or "celular" in label:
        if "TELEFONE" not in labels_ja_tem:
            fields.append(ExtractedField("TELEFONE", value, 0.8))
