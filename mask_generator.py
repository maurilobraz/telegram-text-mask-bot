from dataclasses import dataclass
import re
from ocr_processor import ExtractionResult


@dataclass
class MaskTemplate:
    name: str
    template: str


PREDEFINED_MASKS = {
    "reagendamento": MaskTemplate(
        name="Reagendamento 7017",
        template="""MASCARA PARA CASOS DE REAGENDAMENTO 7017 (o cliente quer para outro dia)

Numero do SA:               {NUMERO_SA}
Matricula do tecnico:       {MATRICULA}
Nome do tecnico:            {NOME_TECNICO}
Motivo da Pendencia:        {MOTIVO}
Nome recebeu o tecnico:     {NOME_RECEBEU}
Contato de quem recebeu:    {CONTATO}
GA Confirmou com o cliente a nova data?
Nome do GA:                 {NOME_GA}
Foto georreferenciada na casa do cliente:""",
    ),
    "simples": MaskTemplate(
        name="Mascara Simples",
        template="""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INFORMACOES EXTRAIDAS DO PRINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{campos_listados}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""",
    ),
}


def generate_mask(result: ExtractionResult, mask_name: str = "reagendamento") -> str:
    """Gera uma mascara preenchida com os dados extraidos."""
    mask = PREDEFINED_MASKS.get(mask_name)
    if not mask:
        mask = PREDEFINED_MASKS["simples"]

    if mask_name == "simples":
        campos = []
        for field in result.fields:
            campos.append(f"  {field.label:<20} | {field.value}")
        if not campos:
            campos.append("  Nenhum campo estruturado identificado.")
        campos_listados = "\n".join(campos)
        return mask.template.format(campos_listados=campos_listados)

    # Monta dict com label -> valor
    field_dict = {f.label: f.value for f in result.fields}

    # Preenche a mascara
    filled = mask.template
    for label, value in field_dict.items():
        placeholder = "{" + label + "}"
        if placeholder in filled:
            filled = filled.replace(placeholder, value)

    # Limpa placeholders nao preenchidos
    filled = re.sub(r"\{[A-Z_]+\}", "____________", filled)

    return filled


def generate_raw_text_mask(result: ExtractionResult) -> str:
    """Retorna o texto bruto extraido formatado."""
    header = "═══════════════════════════════════\n"
    header += "   TEXTO EXTRAIDO (OCR RAW)\n"
    header += "═══════════════════════════════════\n\n"
    return header + result.raw_text


def list_available_masks() -> str:
    """Lista as mascaras disponiveis."""
    lines = ["Mascaras disponiveis:\n"]
    for key, mask in PREDEFINED_MASKS.items():
        lines.append(f"  /mask_{key}  ->  {mask.name}")
    lines.append(f"\n  /raw  ->  Texto bruto do OCR")
    return "\n".join(lines)
