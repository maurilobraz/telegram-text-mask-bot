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
    "Reagendamento - cliente quer outro dia",
    "Reagendamento - cliente nao esta em casa",
    "Reagendamento - problema no equipamento",
    "Reagendamento - falta de peca",
    "Reagendamento - cliente desistiu",
    "Reagendamento - horario nao conveniente",
    "Outro motivo",
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
    return data.get("motivos", DEFAULT_MOTIVOS.copy())


def add_motivo(user_id: int, motivo: str):
    data = load_user_data(user_id)
    motivos = data.get("motivos", DEFAULT_MOTIVOS.copy())
    if motivo not in motivos and motivo != "OUTRO MOTIVO":
        motivos.insert(0, motivo)
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
        buttons.append([InlineKeyboardButton(m, callback_data=f"motivo_{i}")])
    return InlineKeyboardMarkup(buttons)


# ─── /start ────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id

    if is_registered(user_id):
        tech = get_tech_info(user_id)
        await update.message.reply_text(
            f"Ola {tech['nome_tecnico']}!\n\n"
            f"Matricula: {tech['matricula']}\n"
            f"GA: {tech['nome_ga']}\n\n"
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
        f"Nome: {data['nome_tecnico']}\n"
        f"GA: {data['nome_ga']}\n\n"
        "Envie um print/screenshot para comecar."
    )
    return ConversationHandler.END


# ─── EDITAR ────────────────────────────────────────────────────
async def editar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tech = get_tech_info(update.message.from_user.id)
    await update.message.reply_text(
        f"Atuais:\nMatricula: {tech['matricula']}\n"
        f"Nome: {tech['nome_tecnico']}\nGA: {tech['nome_ga']}\n\n"
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


# ─── CALLBACK BOTOES (motivos) ────────────────────────────────
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    if data.startswith("motivo_"):
        idx = int(data[6:])
        motivos = get_motivos(user_id)
        if idx < len(motivos):
            motivo = motivos[idx]
            if motivo == "Outro motivo":
                await query.edit_message_text("Digite o motivo:")
                context.user_data["aguardando_motivo_custom"] = True
            else:
                context.user_data["motivo_selecionado"] = motivo
                await query.edit_message_text(
                    "Qual o nome da pessoa que recebeu o tecnico?"
                )
                context.user_data["aguardando_nome_recebeu"] = True


# ─── TEXTO LIVRE ──────────────────────────────────────────────
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    text = update.message.text.strip()

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
        await update.message.reply_text(
            "Dados completos!\n\n"
            "Use /reagendamento para gerar a mascara."
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
        "MATRICULA": tech["matricula"],
        "NOME_TECNICO": tech["nome_tecnico"],
        "NOME_GA": tech["nome_ga"],
    }

    mask_text = generate_mask(result, "reagendamento", extra)
    await update.message.reply_text(mask_text)


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
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Bot iniciado!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
