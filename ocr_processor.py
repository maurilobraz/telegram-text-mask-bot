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


LABEL_ALIASES = {
    "numero do sa": "NUMERO_SA",
    "sa": "NUMERO_SA",
    "n do sa": "NUMERO_SA",
    "nº sa": "NUMERO_SA",
    "n. sa": "NUMERO_SA",
    "num sa": "NUMERO_SA",
    "n° sa": "NUMERO_SA",
    "tipo de atividade": "ATIVIDADE",
    "atividade": "ATIVIDADE",
    "tipo atividade": "ATIVIDADE",
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
    "titular": "NOME_CLIENTE",
    "telefone": "TELEFONE",
    "celular": "TELEFONE",
    "tel": "TELEFONE",
    "cel": "TELEFONE",
    "data": "DATA",
    "hora": "HORA",
    "email": "EMAIL",
    "e-mail": "EMAIL",
}


def load_and_prep(image_bytes: bytes) -> np.ndarray:
    """Carrega e prepara a imagem."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Nao foi possivel carregar a imagem")
    return img


def enhance_for_ocr(img: np.ndarray) -> list[np.ndarray]:
    """Cria multiplas versoes da imagem para melhor deteccao."""
    versions = []

    # Original
    versions.append(img)

    # Cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    versions.append(gray)

    # CLAHE (contraste local adaptativo)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    versions.append(cl)

    # Threshold OTSU
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(otsu)

    # Threshold OTSU invertido
    versions.append(cv2.bitwise_not(otsu))

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    versions.append(adaptive)

    # Binarizacao com limiar fixo (bom pra texto preto em fundo claro)
    _, fixed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    versions.append(fixed)

    # Morphological cleanup
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    versions.append(cleaned)

    return versions


def upscale(img: np.ndarray, factor: int = 3) -> np.ndarray:
    """Aumenta a imagem para melhor OCR."""
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def extract_sa_from_yellow_region(image_bytes: bytes) -> list[str]:
    """Extrai SA especificamente da regiao amarela no topo."""
    img = load_and_prep(image_bytes)
    h, w = img.shape[:2]

    # Topo da imagem (onde fica o SA com fundo amarelo)
    top = img[0:int(h * 0.30), :]

    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Dilata para pegar a regiao completa
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    yellow_only = cv2.bitwise_and(top, top, mask=mask)
    gray = cv2.cvtColor(yellow_only, cv2.COLOR_BGR2GRAY)

    # Multiplas versoes
    versions = []
    for thresh_val in [0, 80, 100, 120, 140]:
        if thresh_val == 0:
            _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, t = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        versions.append(t)
        versions.append(cv2.bitwise_not(t))

    results = []
    configs = [r"--oem 3 --psm 7", r"--oem 3 --psm 8", r"--oem 3 --psm 13"]

    for v in versions:
        big = upscale(v, 4)
        # Remove ruido
        big = cv2.medianBlur(big, 3)
        for config in configs:
            try:
                pil = Image.fromarray(big)
                text = pytesseract.image_to_string(pil, lang="por", config=config)
                text = text.strip()
                if text:
                    results.append(text)
            except Exception:
                continue

    return results


def extract_text_multi_pass(image_bytes: bytes, lang: str = "por") -> str:
    """Extrai texto com multiplas passadas."""
    img = load_and_prep(image_bytes)
    versions = enhance_for_ocr(img)

    all_texts = []
    configs = [
        r"--oem 3 --psm 6",
        r"--oem 3 --psm 4",
        r"--oem 3 --psm 3",
        r"--oem 3 --psm 11",
        r"--oem 3 --psm 12",
    ]

    for v in versions:
        # Versao normal
        for config in configs:
            try:
                pil = Image.fromarray(v)
                text = pytesseract.image_to_string(pil, lang=lang, config=config)
                if text.strip():
                    all_texts.append(text)
            except Exception:
                continue

        # Versao aumentada (só pra imagens pequenas)
        h, w = v.shape[:2] if len(v.shape) == 3 else v.shape
        if max(h, w) < 2000:
            big = upscale(v, 2)
            for config in configs[:2]:
                try:
                    pil = Image.fromarray(big)
                    text = pytesseract.image_to_string(pil, lang=lang, config=config)
                    if text.strip():
                        all_texts.append(text)
                except Exception:
                    continue

    return "\n".join(all_texts)


def extract_sa_number(text: str) -> str:
    """Busca numero de SA (8 digitos) no texto."""
    patterns = [
        # SA seguido de 8 digitos
        r'(?:SA|S\.A\.?)\s*[:;\-=\s]\s*(\d{8})',
        # N SA ou Nº SA
        r'(?:N[ºo°.]?\s*(?:do\s+)?SA)\s*[:;\-=\s]\s*(\d{8})',
        # Qualquer coisa seguida de 8 digitos no contexto de SA
        r'SA\s*(\d{8})',
        # 8 digitos isolados (menos confiavel)
        r'\b(\d{8})\b',
    ]
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        if matches:
            return matches[0]
    return ""


def extract_phone_number(text: str) -> str:
    """Busca telefone brasileiro."""
    patterns = [
        r'\(\d{2}\)\s*\d{4,5}[-.\s]?\d{4}',
        r'\d{2}\s*\d{4,5}[-.\s]?\d{4}',
        r'\(\d{2}\)\s*\d{8,9}',
        r'\d{10,11}',
    ]
    for p in patterns:
        matches = re.findall(p, text)
        for m in matches:
            # Valida se parece telefone real
            digits = re.sub(r'\D', '', m)
            if 10 <= len(digits) <= 11:
                return m.strip()
    return ""


def extract_address(text: str) -> str:
    """Busca endereco."""
    keywords = [
        "rua", "av.", "avenida", "alameda", "travessa", "beco",
        "praca", "estrada", "rodovia", "bairro", "cidade", "cep",
        "lote", "quadra", "conjunto", "residencial"
    ]
    lines = text.split("\n")
    for line in lines:
        line_lower = line.lower().strip()
        for kw in keywords:
            if kw in line_lower and len(line.strip()) > 10:
                return line.strip()
    return ""


def extract_client_name(text: str) -> str:
    """Busca nome do cliente."""
    patterns = [
        r'(?:cliente|nome\s+do\s+cliente|titular)\s*[:;=]\s*(.+)',
        r'(?:nome)\s*[:;=]\s*([A-ZÁÉÍÓÚÃÕÊÔ][a-záéíóúãõêô]+(?:\s+[A-ZÁÉÍÓÚÃÕÊÔ][a-záéíóúãõêô]+)+)',
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def extract_atividade(text: str) -> str:
    """Busca tipo de atividade."""
    patterns = [
        r'(?:tipo\s+(?:de\s+)?atividade|atividade)\s*[:;=]\s*(.+)',
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def normalize_label(label: str) -> str:
    """Normaliza o label."""
    label_lower = label.lower().strip()
    label_lower = re.sub(r'^[\s\-–—]+|[\s\-–—]+$', '', label_lower)
    return LABEL_ALIASES.get(label_lower, label.upper().replace(" ", "_"))


def identify_fields(text: str) -> list[ExtractedField]:
    """Identifica campos no texto."""
    fields = []
    seen = set()

    # SA (8 digitos)
    sa = extract_sa_number(text)
    if sa:
        fields.append(ExtractedField("NUMERO_SA", sa, 0.95))
        seen.add("NUMERO_SA")

    # Telefone
    phone = extract_phone_number(text)
    if phone:
        fields.append(ExtractedField("TELEFONE", phone, 0.9))
        seen.add("TELEFONE")

    # Endereco
    addr = extract_address(text)
    if addr:
        fields.append(ExtractedField("ENDERECO", addr, 0.8))
        seen.add("ENDERECO")

    # Nome cliente
    name = extract_client_name(text)
    if name:
        fields.append(ExtractedField("NOME_CLIENTE", name, 0.75))
        seen.add("NOME_CLIENTE")

    # Atividade
    atividade = extract_atividade(text)
    if atividade:
        fields.append(ExtractedField("ATIVIDADE", atividade, 0.8))
        seen.add("ATIVIDADE")

    # Campos label: valor
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

            if norm_label in seen:
                continue
            seen.add(norm_label)

            fields.append(ExtractedField(norm_label, value, 0.85))

    return fields


def process_image(image_bytes: bytes, lang: str = "por") -> ExtractionResult:
    """Processa imagem com OCR maximo."""
    # 1. Texto geral multi-pass
    raw_text = extract_text_multi_pass(image_bytes, lang)

    # 2. SA da regiao amarela
    sa_texts = extract_sa_from_yellow_region(image_bytes)

    # 3. Campos do texto geral
    fields = identify_fields(raw_text)

    # 4. Melhora SA se achou na regiao amarela
    for sa_text in sa_texts:
        sa_num = extract_sa_number(sa_text)
        if sa_num and len(sa_num) == 8:
            found = False
            for f in fields:
                if f.label == "NUMERO_SA":
                    f.value = sa_num
                    f.confidence = 0.98
                    found = True
                    break
            if not found:
                fields.insert(0, ExtractedField("NUMERO_SA", sa_num, 0.98))
            break

    # 5. Se nao achou SA ainda, tenta nas primeiras linhas
    if not any(f.label == "NUMERO_SA" for f in fields):
        lines = raw_text.split("\n")
        for line in lines[:15]:
            sa = extract_sa_number(line)
            if sa:
                fields.insert(0, ExtractedField("NUMERO_SA", sa, 0.9))
                break

    return ExtractionResult(raw_text=raw_text, fields=fields)
