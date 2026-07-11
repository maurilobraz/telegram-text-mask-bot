import pytesseract
import cv2
import numpy as np
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


# Labels do OCR
LABEL_ALIASES = {
    "numero do sa": "NUMERO_SA",
    "sa": "NUMERO_SA",
    "n do sa": "NUMERO_SA",
    "nº sa": "NUMERO_SA",
    "n. sa": "NUMERO_SA",
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
    "endereco": "ENDERECO",
    "endereço": "ENDERECO",
    "endereco do cliente": "ENDERECO",
    "nome do cliente": "NOME_CLIENTE",
    "cliente": "NOME_CLIENTE",
    "telefone": "TELEFONE",
    "celular": "TELEFONE",
    "tel": "TELEFONE",
    "cel": "TELEFONE",
    "data": "DATA",
    "hora": "HORA",
    "email": "EMAIL",
    "e-mail": "EMAIL",
}


def preprocess_image(image_bytes: bytes) -> list[np.ndarray]:
    """Preprocessa a imagem em varias versoes para melhor OCR."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    versions = []

    # 1. Original
    versions.append(("original", img))

    # 2. Cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    versions.append(("gray", gray))

    # 3. Alto contraste (preto e branco limpo)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(("thresh", thresh))

    # 4. Invertido (util pra fundo claro com texto escuro)
    inv = cv2.bitwise_not(thresh)
    versions.append(("inverted", inv))

    # 5. Regiao amarela (onde fica o SA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_region = cv2.bitwise_and(img, img, mask=mask)
    gray_yellow = cv2.cvtColor(yellow_region, cv2.COLOR_BGR2GRAY)
    _, thresh_yellow = cv2.threshold(gray_yellow, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(("yellow_region", thresh_yellow))

    return versions


def extract_text_advanced(image_bytes: bytes, lang: str = "por") -> str:
    """Extrai texto usando multiplas versoes da imagem."""
    versions = preprocess_image(image_bytes)
    all_texts = []

    configs = [
        r"--oem 3 --psm 6",
        r"--oem 3 --psm 4",
        r"--oem 3 --psm 3",
    ]

    for name, img in versions:
        for config in configs:
            try:
                pil_img = Image.fromarray(img)
                text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
                if text.strip():
                    all_texts.append(text)
            except Exception:
                continue

    # Junta todos os textos unicos
    combined = "\n".join(all_texts)
    return combined


def extract_sa_from_yellow(image_bytes: bytes) -> str:
    """Tenta extrair especificamente o numero SA da regiao amarela."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Pega so a parte de cima (onde fica o SA)
    h, w = img.shape[:2]
    top_region = img[0:int(h * 0.25), :]

    # Detecta regiao amarela
    hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Aplica mascara
    yellow_only = cv2.bitwise_and(top_region, top_region, mask=mask)

    # Converte pra cinza e aplica threshold
    gray = cv2.cvtColor(yellow_only, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Aumenta a imagem pra OCR pegar melhor
    scale = 3
    big = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    pil_img = Image.fromarray(big)
    text = pytesseract.image_to_string(pil_img, lang="por", config=r"--oem 3 --psm 7")
    return text.strip()


def normalize_label(label: str) -> str:
    """Normaliza o label."""
    label_lower = label.lower().strip()
    label_lower = re.sub(r'^[\s\-–—]+|[\s\-–—]+$', '', label_lower)
    return LABEL_ALIASES.get(label_lower, label.upper().replace(" ", "_"))


def extract_sa_from_text(text: str) -> str:
    """Busca numero de SA no texto."""
    # Padroes comuns de SA
    patterns = [
        r'(?:SA|N[ºo°.]?\s*SA|N[ºo°.]?\s*S\.?A\.?)\s*[:;=\-]?\s*(\d{5,10})',
        r'(?:SA|N[ºo°.]?\s*SA)\s*[:;=\-]?\s*(\d+)',
        r'\b(\d{8,10})\b',  # Numeros grandes provaveis de SA
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1) if match.lastindex else match.group(0)
    return ""


def extract_phone(text: str) -> str:
    """Busca telefone no texto."""
    patterns = [
        r'\(?\d{2}\)?\s*\d{4,5}[-.]?\d{4}',
        r'\d{2}\s*\d{4,5}[-.]?\d{4}',
    ]
    for p in patterns:
        match = re.search(p, text)
        if match:
            return match.group(0).strip()
    return ""


def extract_address(text: str) -> str:
    """Tenta extrair endereco do texto."""
    # Busca por linhas que parecem endereco
    addr_keywords = ["rua", "av.", "avenida", "alameda", "travessa", "bairro", "cidade", "cep"]
    for line in text.split("\n"):
        line_lower = line.lower().strip()
        for kw in addr_keywords:
            if kw in line_lower and len(line.strip()) > 10:
                return line.strip()
    return ""


def identify_fields(text: str) -> list[ExtractedField]:
    """Identifica campos no texto."""
    fields = []
    seen_labels = set()

    # Primeiro: busca SA e telefone com padroes especificos
    sa = extract_sa_from_text(text)
    if sa:
        fields.append(ExtractedField(label="NUMERO_SA", value=sa, confidence=0.95))
        seen_labels.add("NUMERO_SA")

    phone = extract_phone(text)
    if phone and "TELEFONE" not in seen_labels:
        fields.append(ExtractedField(label="TELEFONE", value=phone, confidence=0.9))
        seen_labels.add("TELEFONE")

    addr = extract_address(text)
    if addr and "ENDERECO" not in seen_labels:
        fields.append(ExtractedField(label="ENDERECO", value=addr, confidence=0.7))
        seen_labels.add("ENDERECO")

    # Depois: campos label: valor
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(.+?)\s*[:;=]\s*(.+)$', line)
        if match:
            raw_label = match.group(1).strip()
            value = match.group(2).strip()

            if len(raw_label) < 2 or len(value) < 1:
                continue

            norm_label = normalize_label(raw_label)

            if norm_label in seen_labels:
                continue
            seen_labels.add(norm_label)

            fields.append(ExtractedField(
                label=norm_label,
                value=value,
                confidence=0.85
            ))

    return fields


def process_image(image_bytes: bytes, lang: str = "por") -> ExtractionResult:
    """Processa imagem com OCR avancado."""
    # Texto avancado com multiplas versoes
    raw_text = extract_text_advanced(image_bytes, lang=lang)

    # Tenta extrair SA especifico da regiao amarela
    sa_yellow = extract_sa_from_yellow(image_bytes)

    # Identifica campos
    fields = identify_fields(raw_text)

    # Se achou SA na regiao amarela e e melhor que o do texto geral
    if sa_yellow:
        sa_clean = extract_sa_from_text(sa_yellow)
        if sa_clean:
            # Atualiza ou adiciona
            found = False
            for f in fields:
                if f.label == "NUMERO_SA":
                    f.value = sa_clean
                    f.confidence = 0.98
                    found = True
                    break
            if not found:
                fields.insert(0, ExtractedField(
                    label="NUMERO_SA", value=sa_clean, confidence=0.98
                ))

    # Se nao achou SA, tenta no topo do texto
    if not any(f.label == "NUMERO_SA" for f in fields):
        lines = raw_text.split("\n")
        for line in lines[:10]:  # So as 10 primeiras linhas
            sa = extract_sa_from_text(line)
            if sa:
                fields.insert(0, ExtractedField(
                    label="NUMERO_SA", value=sa, confidence=0.9
                ))
                break

    return ExtractionResult(raw_text=raw_text, fields=fields)
