import pytesseract
from PIL import Image
import re
import io
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


# Mapeamento de labels do OCR para nomes padrao do sistema
LABEL_ALIASES = {
    "numero do sa": "NUMERO_SA",
    "sa": "NUMERO_SA",
    "matricula do tecnico": "MATRICULA",
    "matricula": "MATRICULA",
    "nome do tecnico": "NOME_TECNICO",
    "tecnico": "NOME_TECNICO",
    "motivo da pendencia": "MOTIVO",
    "motivo": "MOTIVO",
    "nome recebeu o tecnico": "NOME_RECEBEU",
    "recebeu": "NOME_RECEBEU",
    "contato de quem recebeu": "CONTATO",
    "contato": "CONTATO",
    "nome do ga": "NOME_GA",
    "ga": "NOME_GA",
    "data": "DATA",
    "hora": "HORA",
    "telefone": "TELEFONE",
    "celular": "TELEFONE",
    "email": "EMAIL",
    "e-mail": "EMAIL",
}


def extract_text_from_image(image_bytes: bytes, lang: str = "por") -> str:
    """Extrai texto de uma imagem usando Tesseract OCR."""
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != "RGB":
        image = image.convert("RGB")

    custom_config = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
    return text


def normalize_label(label: str) -> str:
    """Normaliza o label para o formato padrao."""
    label_lower = label.lower().strip()
    # Remove caracteres especiais no inicio/fim
    label_lower = re.sub(r'^[\s\-–—]+|[\s\-–—]+$', '', label_lower)
    return LABEL_ALIASES.get(label_lower, label.upper().replace(" ", "_"))


def identify_fields(text: str) -> list[ExtractedField]:
    """Identifica campos no formato label: valor no texto extraido."""
    fields = []
    seen_labels = set()

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Busca por padrao: label : valor ou label = valor
        match = re.match(r'^(.+?)\s*[:;=]\s*(.+)$', line)
        if match:
            raw_label = match.group(1).strip()
            value = match.group(2).strip()

            # Ignora linhas muito curtas ou sem valor
            if len(raw_label) < 2 or len(value) < 1:
                continue

            norm_label = normalize_label(raw_label)

            # Evita duplicatas
            if norm_label in seen_labels:
                continue
            seen_labels.add(norm_label)

            fields.append(ExtractedField(
                label=norm_label,
                value=value,
                confidence=0.9
            ))

    return fields


def process_image(image_bytes: bytes, lang: str = "por") -> ExtractionResult:
    """Processa uma imagem: extrai texto e identifica campos."""
    raw_text = extract_text_from_image(image_bytes, lang=lang)
    fields = identify_fields(raw_text)
    return ExtractionResult(raw_text=raw_text, fields=fields)
