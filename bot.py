import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ConversationHandler,
    filters,
    ContextTypes,
)

from ocr_processor import process_image
from mask_generator import generate_mask, generate_raw_text_mask

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MATRICULA, NOME_TECNICO, NOME_GA = range(3)

DATA_DIR = Path("user_data")
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_MOTIVOS = [
    "REAGENDAMENTO - CLIENTE QUER OUTRO DIA",
    "REAGENDAMENTO - CLIENTE NAO ESTA EM CASA",
    "REAGENDAMENTO - PROBLEMA NO EQUIPAMENTO",
    "REAGENDAMENTO - FALTA DE PECA",
    "REAGENDAMENTO - CLIENTE DESISTIU",
    "REAGENDAMENTO - HORARIO NAO CONVENIENTE",
    "OUTRO MOTIVO",
]


def get_user_file(user_id: int) -> Path:
    return DATA_DIR / f"{user_id}.json"


def load_user_data(user_id: int) -> dict:
    f = get_user_file(user_id)
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {}


def save_user_data(user_id: int, data: dict):
    f = get_user_file(user_id)
    f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_motivos(user_id: int) -> list[str]:
    data = load_user_data(user_id)
    motivos = data.get("motivos", DEFAULT_MOTIVOS.copy())
    return [m.upper() for m in motivos]


def add_motivo(user_id: int, motivo: str):
    data = load_user_data(user_id)
    motivos = data.get("motivos", DEFAULT_MOTIVOS.copy())
    motivo_upper = motivo.upper()
    if motivo_upper not in [m.upper() for m in motivos] and motivo_upper != "OUTRO MOTIVO":
        motivos.insert(0, motivo_upper)
    data["motivos"] = motivos
    save_user_data(user_id, data)


def get_tech_info(user_id: int) -> dict:
    data = load_user_data(user_id)
    return {
        "matricula": data.get("matricula", ""),
        "nome_tecnico": data.get("nome_tecnico", ""),
        "nome_ga": data.get("nome_ga", ""),
    }


def is_registered(user_id: int) -> bool:
    data = load_user_data(user_id)
    return bool(data.get("matricula"))


def get_motivo_keyboard(user_id: int) -> InlineKeyboardMarkup:
    motivos = get_motivos(user_id)
    buttons = []
    for i, m in enumerate(motivos):
        buttons.append([InlineKeyboardButton(m.title(), callback_data=f"motivo_{i}")])
    return InlineKeyboardMarkup(buttons)


def get_nome_keyboard(cliente_nome: str) -> InlineKeyboardMarkup:
    buttons = []
    if cliente_nome:
        buttons.append([
            InlineKeyboardButton(
                f"Titular - {cliente_nome.title()}",
                callback_data="nome_titular"
            )
        ])
    buttons.append([InlineKeyboardButton("Digitar outro nome", callback_data="nome_outro")])
    return InlineKeyboardMarkup(buttons)


def get_contato_keyboard(contato: str) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(f"Sim, usar {contato}", callback_data="contato_sim")],
        [InlineKeyboardButton("Nao, digitar outro numero", callback_data="contato_nao")],
    ]
    return InlineKeyboardMarkup(buttons)


def get_ga_confirmou_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("Sim", callback_data="ga_sim")],
        [InlineKeyboardButton("Nao", callback_data="ga_nao")],
    ]
    return InlineKeyboardMarkup(buttons)


def get_send_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("Enviar para grupo", callback_data="send_group")],
        [InlineKeyboardButton("Copiar texto", callback_data="copy_text")],
    ]
    return InlineKeyboardMarkup(buttons)


def get_second_print_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("Enviar outro print", callback_data="wait_second")],
        [InlineKeyboardButton("Usar esses dados", callback_data="skip_second")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_extra(ud: dict, tech: dict) -> dict:
    return {
        "SA": ud.get("sa_extraido", ""),
        "IDCOMPANHIA": ud.get("idcompanhia_extraido", ""),
        "ATIVIDADE": ud.get("atividade_extraida", ""),
        "CONTATO_OCR": ud.get("contato_extraido", ""),
        "MOTIVO": ud.get("motivo_selecionado", ""),
        "OBSERVACAO": ud.get("observacao", ""),
        "NOME_RECEBEU": ud.get("nome_recebeu", ""),
        "CONTATO_CLIENTE": ud.get("contato_cliente", ""),
        "NOME_CLIENTE": ud.get("cliente_nome", ""),
        "ENDERECO": ud.get("endereco_extraido", ""),
        "TELEFONE": ud.get("contato_extraido", ""),
        "GA_CONFIRMOU": ud.get("ga_confirmou", ""),
        "MATRICULA": tech["matricula"],
        "NOME_TECNICO": tech["nome_tecnico"],
        "NOME_GA": tech["nome_ga"],
    }


CAMPOS_LABELS = {
    "sa_extraido": "NUMERO_SA",
    "cliente_nome": "NOME DO CLIENTE",
    "endereco_extraido": "ENDERECO DO CLIENTE",
    "motivo_selecionado": "MOTIVO",
    "observacao": "OBSERVACAO",
    "nome_recebeu": "NOME QUE RECEBEU O TECNICO",
    "contato_cliente": "CONTATO DE QUEM RECEBEU",
    "ga_confirmou": "GA CONFIRMOU",
}


def get_campos_faltando(ud: dict) -> list[str]:
    """Retorna lista de campos que ainda nao foram preenchidos."""
    from mask_generator import PREDEFINED_MASKS
    mask_name = ud.get("mascara_selecionada", "reagendamento")
    mask = PREDEFINED_MASKS.get(mask_name)
    required = mask.required_fields if mask and mask.required_fields else list(CAMPOS_LABELS.keys())

    faltando = []
    for key in required:
        if not ud.get(key):
            faltando.append((key, CAMPOS_LABELS.get(key, key)))
    return faltando


async def perguntar_proximo_campo(update, context, user_id):
    """Pergunta o proximo campo que falta preencher."""
    faltando = get_campos_faltando(context.user_data)

    if not faltando:
        # Todos preenchidos - gera mascara
        if "ocr_results" in context.user_data and context.user_data["ocr_results"]:
            result = context.user_data["ocr_results"][-1]
            tech = get_tech_info(user_id)
            extra = build_extra(context.user_data, tech)
            mask_text = generate_mask(result, "reagendamento", extra)
            context.user_data["ultimo_texto_mascara"] = mask_text

            await update.message.reply_text(
                mask_text,
                reply_markup=get_send_keyboard()
            )
            return True
        else:
            await update.message.reply_text("Envie um print primeiro.")
            return True

    # Pergunta o proximo campo
    key, label = faltando[0]

    if key == "sa_extraido":
        await update.message.reply_text("Qual o numero do SA? (8 digitos)")
        context.user_data["aguardando_sa"] = True

    elif key == "atividade_extraida":
        await update.message.reply_text("Qual o tipo de atividade?")
        context.user_data["aguardando_atividade"] = True

    elif key == "cliente_nome":
        await update.message.reply_text("Qual o nome do cliente (titular)?")
        context.user_data["aguardando_cliente_nome"] = True

    elif key == "endereco_extraido":
        await update.message.reply_text("Qual o endereco do cliente?")
        context.user_data["aguardando_endereco"] = True

    elif key == "motivo_selecionado":
        await update.message.reply_text(
            "Qual o motivo da pendencia?",
            reply_markup=get_motivo_keyboard(user_id)
        )

    elif key == "observacao":
        await update.message.reply_text("Qual a observacao?")
        context.user_data["aguardando_observacao"] = True

    elif key == "nome_recebeu":
        cliente_nome = context.user_data.get("cliente_nome", "")
        await update.message.reply_text(
            "Qual o nome da pessoa que recebeu o tecnico?",
            reply_markup=get_nome_keyboard(cliente_nome)
        )
        context.user_data["aguardando_escolha_nome"] = True

    elif key == "contato_cliente":
        contato = context.user_data.get("contato_extraido", "")
        if contato:
            await update.message.reply_text(
                f"O numero extraido e: {contato}\nUsar este numero?",
                reply_markup=get_contato_keyboard(contato)
            )
            context.user_data["aguardando_escolha_contato"] = True
        else:
            await update.message.reply_text("Qual o numero de contato?")
            context.user_data["aguardando_contato_cliente"] = True

    elif key == "ga_confirmou":
        await update.message.reply_text(
            "GA confirmou com o cliente a nova data?",
            reply_markup=get_ga_confirmou_keyboard()
        )
        context.user_data["aguardando_ga"] = True

    return False


# ─── /start ────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id

    context.user_data["print1_recebido"] = False
    context.user_data["print2_recebido"] = False
    context.user_data["prints_processados"] = False

    if is_registered(user_id):
        tech = get_tech_info(user_id)
        await update.message.reply_text(
            f"Ola {tech['nome_tecnico'].title()}!\n\n"
            f"Matricula: {tech['matricula']}\n"
            f"GA: {tech['nome_ga'].title()}\n\n"
            "Envie os prints:\n"
            "1. Print com SA, atividade e nome do cliente\n"
            "2. Print com contato\n\n"
            "Depois escolha a mascara:\n"
            "/reagendamento - Reagendamento 7017\n"
            "/cdoe - CDOE SEM POTENCIA"
        )
    else:
        await update.message.reply_text(
            "Bem-vindo! Preciso dos seus dados para configurar.\n\n"
            "Qual sua matricula? (ex: TT821674)"
        )
        return MATRICULA
    return ConversationHandler.END


# ─── CADASTRO ─────────────────────────────────────────────────
async def ask_matricula(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    matricula = update.message.text.strip().upper()
    
    # Valida formato: 2 letras + 6 numeros (ex: TT577337)
    import re
    if not re.match(r'^[A-Z]{2}\d{6}$', matricula):
        await update.message.reply_text(
            "Formato invalido! A matricula deve ter 2 letras seguidas de 6 numeros.\n"
            "Exemplo: TT577337\n\n"
            "Qual sua matricula?"
        )
        return MATRICULA
    
    context.user_data["temp_matricula"] = matricula
    await update.message.reply_text("Qual seu nome completo?")
    return NOME_TECNICO


async def ask_nome_tecnico(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    nome = update.message.text.strip()
    
    # Valida pelo menos 2 nomes
    palavras = [p for p in nome.split() if len(p) >= 2]
    if len(palavras) < 2:
        await update.message.reply_text(
            "Nome invalido! Precisa ter nome e sobrenome.\n"
            "Exemplo: Joao Silva\n\n"
            "Qual seu nome completo?"
        )
        return NOME_TECNICO
    
    context.user_data["temp_nome_tecnico"] = nome.upper()
    await update.message.reply_text("Qual o nome do GA responsavel?")
    return NOME_GA


async def save_cadastro(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    data = load_user_data(user_id)
    data["matricula"] = context.user_data["temp_matricula"]
    data["nome_tecnico"] = context.user_data["temp_nome_tecnico"]
    data["nome_ga"] = update.message.text.strip().upper()
    data["motivos"] = DEFAULT_MOTIVOS.copy()
    save_user_data(user_id, data)

    await update.message.reply_text(
        "Cadastro concluido!\n\n"
        f"Matricula: {data['matricula']}\n"
        f"Nome: {data['nome_tecnico'].title()}\n"
        f"GA: {data['nome_ga'].title()}\n\n"
        "Envie os prints:\n"
        "1. Print com SA, atividade e nome do cliente\n"
        "2. Print com contato"
    )
    return ConversationHandler.END


# ─── EDITAR ────────────────────────────────────────────────────
async def editar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tech = get_tech_info(update.message.from_user.id)
    await update.message.reply_text(
        f"Atuais:\nMatricula: {tech['matricula']}\n"
        f"Nome: {tech['nome_tecnico'].title()}\nGA: {tech['nome_ga'].title()}\n\n"
        "Qual sua nova matricula?"
    )
    return MATRICULA


# ─── PROCESSAR IMAGEM ──────────────────────────────────────────
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id

    if not is_registered(user_id):
        await update.message.reply_text("Primeiro faca seu cadastro com /start")
        return

    await update.message.reply_text("Processando imagem...")

    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()

        result = process_image(bytes(image_bytes))

        # Salva resultado OCR
        if "ocr_results" not in context.user_data:
            context.user_data["ocr_results"] = []
        context.user_data["ocr_results"].append(result)

        # Salva campos no user_data (NAO sobrescreve)
        for f in result.fields:
            if f.label == "SA" and not context.user_data.get("sa_extraido"):
                context.user_data["sa_extraido"] = f.value
            elif f.label == "CLIENTE" and not context.user_data.get("cliente_nome"):
                context.user_data["cliente_nome"] = f.value
            elif f.label == "ENDERECO" and not context.user_data.get("endereco_extraido"):
                context.user_data["endereco_extraido"] = f.value
            elif f.label == "TELEFONE" and not context.user_data.get("contato_extraido"):
                context.user_data["contato_extraido"] = f.value

        # Verifica se achou SA
        if not context.user_data.get("sa_extraido"):
            await update.message.reply_text("Nao encontrei o SA. Envie novamente.")
            return

        # Monta resumo
        resumo = "Dados capturados:\n\n"
        resumo += f"SA: {context.user_data.get('sa_extraido', '-')}\n"
        resumo += f"Cliente: {context.user_data.get('cliente_nome', '-')}\n"
        resumo += f"Endereco: {context.user_data.get('endereco_extraido', '-')}\n"
        resumo += f"Contato: {context.user_data.get('contato_extraido', '-')}\n"

        resumo += "\nEnvie outro print ou clique 'Usar esses dados':"
        await update.message.reply_text(resumo, reply_markup=get_second_print_keyboard())

    except Exception as e:
        logger.error(f"Erro: {e}")
        await update.message.reply_text(f"Erro ao processar: {str(e)}")


# ─── CALLBACK BOTOES ───────────────────────────────────────────
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id

    logger.info(f"Callback: data={data}")

    # ── Segundo print ──
    if data == "wait_second":
        await query.answer()
        await query.edit_message_text("Aguardando o segundo print (contato)...")
        return

    if data == "skip_second":
        await query.answer()
        context.user_data["prints_processados"] = True
        await query.edit_message_text("Verificando dados...")
        await perguntar_proximo_campo(query, context, user_id)
        return

    # ── Motivo ──
    if data.startswith("motivo_"):
        try:
            idx = int(data[7:])
        except ValueError:
            await query.answer("Erro.")
            return

        motivos = get_motivos(user_id)
        if idx < len(motivos):
            motivo = motivos[idx]
            context.user_data["motivo_selecionado"] = motivo

            if motivo.upper() == "OUTRO MOTIVO":
                await query.answer()
                await query.edit_message_text("Digite o motivo:")
                context.user_data["aguardando_motivo_custom"] = True
            else:
                await query.answer()
                cliente_nome = context.user_data.get("cliente_nome", "")
                await query.edit_message_text(
                    f"Motivo: {motivo.title()}\n\n"
                    "Qual o nome da pessoa que recebeu o tecnico?",
                    reply_markup=get_nome_keyboard(cliente_nome)
                )
                context.user_data["aguardando_escolha_nome"] = True
        else:
            await query.answer("Opcao invalida.")
        return

    # ── Nome: TITULAR ──
    if data == "nome_titular":
        cliente_nome = context.user_data.get("cliente_nome", "")
        context.user_data["nome_recebeu"] = cliente_nome.upper()
        await query.answer()

        contato = context.user_data.get("contato_extraido", "")
        if contato:
            await query.edit_message_text(
                f"Nome: {cliente_nome.title()}\n\n"
                f"O numero extraido e: {contato}\n"
                "Usar este numero?",
                reply_markup=get_contato_keyboard(contato)
            )
            context.user_data["aguardando_escolha_contato"] = True
        else:
            await query.edit_message_text(
                f"Nome: {cliente_nome.title()}\n\n"
                "Qual o numero de contato da pessoa?"
            )
            context.user_data["aguardando_contato_cliente"] = True
        return

    # ── Nome: OUTRO ──
    if data == "nome_outro":
        await query.answer()
        await query.edit_message_text("Digite o nome da pessoa:")
        context.user_data["aguardando_nome_recebeu"] = True
        return

    # ── Contato: SIM ──
    if data == "contato_sim":
        contato = context.user_data.get("contato_extraido", "")
        context.user_data["contato_cliente"] = contato
        await query.answer()

        ga = context.user_data.get("ga_confirmou", "")
        await query.edit_message_text(
            f"Contato: {contato}\n\n"
            "GA confirmou com o cliente a nova data?",
            reply_markup=get_ga_confirmou_keyboard()
        )
        context.user_data["aguardando_ga"] = True
        return

    # ── Contato: NAO ──
    if data == "contato_nao":
        await query.answer()
        await query.edit_message_text("Digite o numero de contato:")
        context.user_data["aguardando_contato_cliente"] = True
        return

    # ── GA Confirmou: SIM ──
    if data == "ga_sim":
        context.user_data["ga_confirmou"] = "SIM"
        await query.answer()
        await perguntar_proximo_campo(query, context, user_id)
        return

    # ── GA Confirmou: NAO ──
    if data == "ga_nao":
        context.user_data["ga_confirmou"] = "NAO"
        await query.answer()
        await perguntar_proximo_campo(query, context, user_id)
        return

    # ── Enviar para grupo ──
    if data == "send_group":
        texto = context.user_data.get("ultimo_texto_mascara", "")
        if texto:
            await query.answer("Mensagem pronta para encaminhar!")
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=texto
            )
        else:
            await query.answer("Nao ha dados.")
        return

    # ── Copiar texto ──
    if data == "copy_text":
        texto = context.user_data.get("ultimo_texto_mascara", "")
        if texto:
            await query.answer("Texto copiado!", show_alert=True)
        else:
            await query.answer("Nao ha texto.")
        return


# ─── TEXTO LIVRE ──────────────────────────────────────────────
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    text = update.message.text.strip()

    # ── SA digitado ──
    if context.user_data.get("aguardando_sa"):
        context.user_data["sa_extraido"] = text.upper()
        context.user_data["aguardando_sa"] = False
        await perguntar_proximo_campo(update, context, user_id)
        return

    # ── Atividade digitada ──
    if context.user_data.get("aguardando_atividade"):
        context.user_data["atividade_extraida"] = text.upper()
        context.user_data["aguardando_atividade"] = False
        await perguntar_proximo_campo(update, context, user_id)
        return

    # ── Nome do cliente digitado ──
    if context.user_data.get("aguardando_cliente_nome"):
        context.user_data["cliente_nome"] = text.upper()
        context.user_data["aguardando_cliente_nome"] = False
        await perguntar_proximo_campo(update, context, user_id)
        return

    # ── Endereco digitado ──
    if context.user_data.get("aguardando_endereco"):
        context.user_data["endereco_extraido"] = text.upper()
        context.user_data["aguardando_endereco"] = False
        await perguntar_proximo_campo(update, context, user_id)
        return

    # ── Motivo custom ──
    if context.user_data.get("aguardando_motivo_custom"):
        context.user_data["motivo_selecionado"] = text.upper()
        add_motivo(user_id, text.upper())
        context.user_data["aguardando_motivo_custom"] = False
        await perguntar_proximo_campo(update, context, user_id)
        return

    # ── Nome digitado ──
    if context.user_data.get("aguardando_nome_recebeu"):
        context.user_data["nome_recebeu"] = text.upper()
        context.user_data["aguardando_nome_recebeu"] = False
        await perguntar_proximo_campo(update, context, user_id)
        return

    # ── Contato digitado ──
    if context.user_data.get("aguardando_contato_cliente"):
        context.user_data["contato_cliente"] = text.upper()
        context.user_data["aguardando_contato_cliente"] = False
        await perguntar_proximo_campo(update, context, user_id)
        return

    # ── Observacao digitada ──
    if context.user_data.get("aguardando_observacao"):
        context.user_data["observacao"] = text.upper()
        context.user_data["aguardando_observacao"] = False
        await perguntar_proximo_campo(update, context, user_id)
        return


# ─── COMANDOS ──────────────────────────────────────────────────
async def cmd_reagendamento(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id

    if "ocr_results" not in context.user_data or not context.user_data["ocr_results"]:
        await update.message.reply_text("Envie um print primeiro.")
        return

    context.user_data["mascara_selecionada"] = "reagendamento"

    # Verifica campos faltando
    faltando = get_campos_faltando(context.user_data)
    if faltando:
        await perguntar_proximo_campo(update, context, user_id)
        return

    result = context.user_data["ocr_results"][-1]
    tech = get_tech_info(user_id)
    extra = build_extra(context.user_data, tech)
    mask_text = generate_mask(result, "reagendamento", extra)
    context.user_data["ultimo_texto_mascara"] = mask_text
    await update.message.reply_text(mask_text, reply_markup=get_send_keyboard())


async def cmd_cdoe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id

    if "ocr_results" not in context.user_data or not context.user_data["ocr_results"]:
        await update.message.reply_text("Envie um print primeiro.")
        return

    context.user_data["mascara_selecionada"] = "cdoe"

    # Verifica campos faltando
    faltando = get_campos_faltando(context.user_data)
    if faltando:
        await perguntar_proximo_campo(update, context, user_id)
        return

    result = context.user_data["ocr_results"][-1]
    tech = get_tech_info(user_id)
    extra = build_extra(context.user_data, tech)
    mask_text = generate_mask(result, "cdoe", extra)
    context.user_data["ultimo_texto_mascara"] = mask_text
    await update.message.reply_text(mask_text, reply_markup=get_send_keyboard())


async def cmd_lista(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if "ocr_results" not in context.user_data or not context.user_data["ocr_results"]:
        await update.message.reply_text("Envie um print primeiro.")
        return
    result = context.user_data["ocr_results"][-1]
    mask_text = generate_mask(result, "simples")
    await update.message.reply_text(mask_text)


async def cmd_raw(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if "ocr_results" not in context.user_data or not context.user_data["ocr_results"]:
        await update.message.reply_text("Envie um print primeiro.")
        return
    all_raw = ""
    for r in context.user_data["ocr_results"]:
        all_raw += generate_raw_text_mask(r) + "\n\n"
    if len(all_raw) > 4000:
        all_raw = all_raw[:4000] + "\n\n... (truncado)"
    await update.message.reply_text(all_raw)


# ─── MAIN ─────────────────────────────────────────────────────
async def post_init(application: Application) -> None:
    await application.bot.set_my_commands([
        BotCommand("start", "Iniciar / cadastro"),
        BotCommand("editar", "Editar dados do tecnico"),
        BotCommand("reagendamento", "Gerar mascara reagendamento 7017"),
        BotCommand("cdoe", "Gerar mascara CDOE SEM POTENCIA"),
        BotCommand("lista", "Gerar lista de campos"),
        BotCommand("raw", "Ver texto bruto do OCR"),
    ])


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Defina TELEGRAM_BOT_TOKEN no .env")

    app = Application.builder().token(token).post_init(post_init).build()

    app.add_handler(CallbackQueryHandler(button_callback))

    cadastro_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start), CommandHandler("editar", editar)],
        states={
            MATRICULA: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_matricula)],
            NOME_TECNICO: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_nome_tecnico)],
            NOME_GA: [MessageHandler(filters.TEXT & ~filters.COMMAND, save_cadastro)],
        },
        fallbacks=[CommandHandler("start", start)],
    )

    app.add_handler(cadastro_handler)
    app.add_handler(CommandHandler("reagendamento", cmd_reagendamento))
    app.add_handler(CommandHandler("cdoe", cmd_cdoe))
    app.add_handler(CommandHandler("lista", cmd_lista))
    app.add_handler(CommandHandler("raw", cmd_raw))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Bot iniciado!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
