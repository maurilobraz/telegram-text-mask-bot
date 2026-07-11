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


def get_send_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("Enviar para grupo", callback_data="send_group")],
        [InlineKeyboardButton("Copiar texto", callback_data="copy_text")],
    ]
    return InlineKeyboardMarkup(buttons)


# ─── /start ────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id

    if is_registered(user_id):
        tech = get_tech_info(user_id)
        await update.message.reply_text(
            f"Ola {tech['nome_tecnico'].title()}!\n\n"
            f"Matricula: {tech['matricula']}\n"
            f"GA: {tech['nome_ga'].title()}\n\n"
            "Envie um print/screenshot para comecar."
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
    context.user_data["temp_matricula"] = update.message.text.strip().upper()
    await update.message.reply_text("Qual seu nome completo?")
    return NOME_TECNICO


async def ask_nome_tecnico(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["temp_nome_tecnico"] = update.message.text.strip().upper()
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
        "Envie um print/screenshot para comecar."
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
        context.user_data["ocr_result"] = result

        sa = ""
        contato = ""
        for f in result.fields:
            if f.label == "NUMERO_SA":
                sa = f.value
            if f.label in ("TELEFONE", "CONTATO"):
                contato = f.value

        context.user_data["sa_extraido"] = sa
        context.user_data["contato_extraido"] = contato

        await update.message.reply_text(
            f"SA: {sa or '(nao encontrado)'}\n"
            f"Contato: {contato or '(nao encontrado)'}\n\n"
            "Qual o motivo da pendencia?",
            reply_markup=get_motivo_keyboard(user_id),
        )

    except Exception as e:
        logger.error(f"Erro: {e}")
        await update.message.reply_text(f"Erro ao processar: {str(e)}")


# ─── CALLBACK BOTOES ───────────────────────────────────────────
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id

    logger.info(f"Callback recebido: data={data}, user_id={user_id}")

    if data.startswith("motivo_"):
        try:
            idx = int(data[7:])
        except ValueError:
            await query.answer("Erro ao selecionar motivo.")
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
                await query.edit_message_text(
                    f"Motivo: {motivo.title()}\n\n"
                    "Qual o nome da pessoa que recebeu o tecnico?"
                )
                context.user_data["aguardando_nome_recebeu"] = True
        else:
            await query.answer("Opcao invalida.")
        return

    if data == "send_group":
        if "ocr_result" not in context.user_data:
            await query.answer("Nao ha dados para enviar.")
            return

        result = context.user_data["ocr_result"]
        tech = get_tech_info(user_id)
        extra = {
            "SA": context.user_data.get("sa_extraido", ""),
            "CONTATO_OCR": context.user_data.get("contato_extraido", ""),
            "MOTIVO": context.user_data.get("motivo_selecionado", ""),
            "NOME_RECEBEU": context.user_data.get("nome_recebeu", ""),
            "CONTATO_CLIENTE": context.user_data.get("contato_cliente", ""),
            "MATRICULA": tech["matricula"],
            "NOME_TECNICO": tech["nome_tecnico"],
            "NOME_GA": tech["nome_ga"],
        }

        mask_text = generate_mask(result, "reagendamento", extra)
        context.user_data["ultimo_texto_mascara"] = mask_text

        await query.answer()
        await query.edit_message_text(
            mask_text + "\n\n"
            "Copie o texto acima e envie para o grupo desejado."
        )
        return

    if data == "copy_text":
        texto = context.user_data.get("ultimo_texto_mascara", "")
        if texto:
            await query.answer("Texto copiado!", show_alert=True)
        else:
            await query.answer("Nao ha texto para copiar.")
        return


# ─── TEXTO LIVRE ──────────────────────────────────────────────
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    text = update.message.text.strip()

    logger.info(f"Texto recebido: {text}")

    if context.user_data.get("aguardando_motivo_custom"):
        context.user_data["motivo_selecionado"] = text.upper()
        add_motivo(user_id, text.upper())
        context.user_data["aguardando_motivo_custom"] = False
        await update.message.reply_text("Qual o nome da pessoa que recebeu o tecnico?")
        context.user_data["aguardando_nome_recebeu"] = True
        return

    if context.user_data.get("aguardando_nome_recebeu"):
        context.user_data["nome_recebeu"] = text.upper()
        context.user_data["aguardando_nome_recebeu"] = False
        await update.message.reply_text("Qual o numero de contato da pessoa?")
        context.user_data["aguardando_contato_cliente"] = True
        return

    if context.user_data.get("aguardando_contato_cliente"):
        context.user_data["contato_cliente"] = text.upper()
        context.user_data["aguardando_contato_cliente"] = False

        if "ocr_result" in context.user_data:
            result = context.user_data["ocr_result"]
            tech = get_tech_info(user_id)
            extra = {
                "SA": context.user_data.get("sa_extraido", ""),
                "CONTATO_OCR": context.user_data.get("contato_extraido", ""),
                "MOTIVO": context.user_data.get("motivo_selecionado", ""),
                "NOME_RECEBEU": context.user_data.get("nome_recebeu", ""),
                "CONTATO_CLIENTE": context.user_data.get("contato_cliente", ""),
                "MATRICULA": tech["matricula"],
                "NOME_TECNICO": tech["nome_tecnico"],
                "NOME_GA": tech["nome_ga"],
            }

            mask_text = generate_mask(result, "reagendamento", extra)
            context.user_data["ultimo_texto_mascara"] = mask_text

            await update.message.reply_text(
                mask_text,
                reply_markup=get_send_keyboard(),
            )
        else:
            await update.message.reply_text(
                "Dados completos!\n\n"
                "Envie um print e use /reagendamento para gerar a mascara."
            )
        return


# ─── COMANDOS DE MASCARA ──────────────────────────────────────
async def cmd_reagendamento(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id

    if "ocr_result" not in context.user_data:
        await update.message.reply_text("Envie um print primeiro.")
        return

    result = context.user_data["ocr_result"]
    tech = get_tech_info(user_id)

    extra = {
        "SA": context.user_data.get("sa_extraido", ""),
        "CONTATO_OCR": context.user_data.get("contato_extraido", ""),
        "MOTIVO": context.user_data.get("motivo_selecionado", ""),
        "NOME_RECEBEU": context.user_data.get("nome_recebeu", ""),
        "CONTATO_CLIENTE": context.user_data.get("contato_cliente", ""),
        "MATRICULA": tech["matricula"],
        "NOME_TECNICO": tech["nome_tecnico"],
        "NOME_GA": tech["nome_ga"],
    }

    mask_text = generate_mask(result, "reagendamento", extra)
    context.user_data["ultimo_texto_mascara"] = mask_text
    await update.message.reply_text(mask_text, reply_markup=get_send_keyboard())


async def cmd_lista(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if "ocr_result" not in context.user_data:
        await update.message.reply_text("Envie um print primeiro.")
        return
    result = context.user_data["ocr_result"]
    mask_text = generate_mask(result, "simples")
    await update.message.reply_text(mask_text)


async def cmd_raw(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if "ocr_result" not in context.user_data:
        await update.message.reply_text("Envie um print primeiro.")
        return
    result = context.user_data["ocr_result"]
    raw = generate_raw_text_mask(result)
    if len(raw) > 4000:
        raw = raw[:4000] + "\n\n... (truncado)"
    await update.message.reply_text(raw)


# ─── MAIN ─────────────────────────────────────────────────────
async def post_init(application: Application) -> None:
    await application.bot.set_my_commands([
        BotCommand("start", "Iniciar / cadastro"),
        BotCommand("editar", "Editar dados do tecnico"),
        BotCommand("reagendamento", "Gerar mascara reagendamento 7017"),
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
    app.add_handler(CommandHandler("lista", cmd_lista))
    app.add_handler(CommandHandler("raw", cmd_raw))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Bot iniciado!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
