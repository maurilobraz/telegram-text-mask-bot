from dataclasses import dataclass, field
import re
from ocr_processor import ExtractionResult


@dataclass
class MaskTemplate:
    name: str
    template: str
    required_fields: list[str] = field(default_factory=list)


PREDEFINED_MASKS = {
    "reagendamento": MaskTemplate(
        name="Reagendamento 7017",
        template="""Mascara para casos de reagendamento 7017 (o cliente quer para outro dia)

Numero do SA: {SA}
Matricula do tecnico: {MATRICULA}
Nome do tecnico: {NOME_TECNICO}
Nome do cliente: {NOME_CLIENTE}
Endereco do cliente: {ENDERECO}
Motivo da Pendencia: {MOTIVO}
Nome recebeu o tecnico: {NOME_RECEBEU}
Contato de quem recebeu: {CONTATO_CLIENTE}
GA Confirmou com o cliente a nova data? {GA_CONFIRMOU}
Nome do GA: {NOME_GA}
Foto georreferenciada na casa do cliente:""",
        required_fields=["sa_extraido", "cliente_nome", "endereco_extraido", "motivo_selecionado", "nome_recebeu", "contato_cliente", "ga_confirmou"],
    ),
    "simples": MaskTemplate(
        name="Lista de Campos",
        template="""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Informacoes extraidas do print
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{campos_listados}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""",
    ),
}


def generate_mask(
    result: ExtractionResult,
    mask_name: str = "reagendamento",
    extra: dict[str, str] | None = None,
) -> str:
    """Gera uma mascara preenchida com os dados extraidos."""
    if extra is None:
        extra = {}

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

    # Monta dict com todos os dados
    field_dict = {}
    for field in result.fields:
        field_dict[field.label] = field.value

    # Adiciona dados extras (fixos do tecnico, motivo, etc.)
    field_dict.update(extra)

    # Preenche a mascara
    filled = mask.template
    for label, value in field_dict.items():
        placeholder = "{" + label + "}"
        if placeholder in filled and value:
            filled = filled.replace(placeholder, value)

    # Limpa placeholders nao preenchidos
    filled = re.sub(r"\{[A-Z_]+\}", "____________", filled)

    return filled


def generate_raw_text_mask(result: ExtractionResult) -> str:
    """Retorna o texto bruto extraido formatado."""
    header = "═══════════════════════════════════\n"
    header += "   Texto extraido (OCR raw)\n"
    header += "═══════════════════════════════════\n\n"
    return header + result.raw_text


def list_available_masks() -> str:
    lines = ["Mascaras disponiveis:\n"]
    for key, mask in PREDEFINED_MASKS.items():
        lines.append(f"  /{key}  ->  {mask.name}")
    return "\n".join(lines)
