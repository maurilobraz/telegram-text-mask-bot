import os
import logging
from dotenv import load_dotenv
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from ocr_processor import process_image
from mask_generator import (
    generate_mask,
    generate_raw_text_mask,
    list_available_masks,
    PREDEFINED_MASKS,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

user_data: dict[int, dict] = {}


def get_mask_keyboard() -> InlineKeyboardMarkup:
    """Cria teclado inline com as mascaras disponiveis."""
    buttons = []
    for key, mask in PREDEFINED_MASKS.items():
        buttons.append([InlineKeyboardButton(mask.name, callback_data=f"mask_{key}")])
    buttons.append([InlineKeyboardButton("Ver texto bruto (RAW)", callback_data="mask_raw")])
    return InlineKeyboardMarkup(buttons)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Bot de Mascara de Texto\n\n"
        "Envie um print/screenshot que eu extraio o texto e gero a mascara.\n\n"
        "Mascaras disponiveis:",
        reply_markup=get_mask_keyboard(),
    )


async def masks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Escolha uma mascara:",
        reply_markup=get_mask_keyboard(),
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    await update.message.reply_text("Processando imagem...")

    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()

        result = process_image(bytes(image_bytes))
        user_data[user_id] = {"result": result}

        response = f"Imagem processada!\nCampos encontrados: {len(result.fields)}\n\n"

        if result.fields:
            for f in result.fields[:20]:
                response += f"{f.label}: {f.value}\n"
        else:
            response += "Nenhum campo identificado.\n"

        response += "\nEscolha uma mascara:"
        await update.message.reply_text(response, reply_markup=get_mask_keyboard())

    except Exception as e:
        logger.error(f"Erro: {e}")
        await update.message.reply_text(f"Erro ao processar: {str(e)}")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Trata os cliques nos botoes inline."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    data = query.data

    if data == "mask_raw":
        if user_id not in user_data:
            await query.edit_message_text("Envie um print primeiro.")
            return
        result = user_data[user_id]["result"]
        raw = generate_raw_text_mask(result)
        if len(raw) > 4000:
            raw = raw[:4000] + "\n\n... (truncado)"
        await query.edit_message_text(raw)
        return

    if data.startswith("mask_"):
        mask_name = data[5:]
        if user_id not in user_data:
            await query.edit_message_text("Envie um print primeiro.")
            return
        result = user_data[user_id]["result"]
        mask_text = generate_mask(result, mask_name)
        await query.edit_message_text(mask_text)


async def raw_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id

    if user_id not in user_data:
        await update.message.reply_text("Envie um print primeiro.")
        return

    result = user_data[user_id]["result"]
    raw = generate_raw_text_mask(result)

    if len(raw) > 4000:
        raw = raw[:4000] + "\n\n... (truncado)"

    await update.message.reply_text(raw)


async def post_init(application: Application) -> None:
    commands = [BotCommand("start", "Iniciar")]
    for key, mask in PREDEFINED_MASKS.items():
        commands.append(BotCommand(key, mask.name))
    commands.append(BotCommand("masks", "Ver mascaras disponiveis"))
    commands.append(BotCommand("raw", "Texto bruto OCR"))
    await application.bot.set_my_commands(commands)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Defina TELEGRAM_BOT_TOKEN no .env")

    app = Application.builder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("masks", masks_cmd))
    app.add_handler(CommandHandler("raw", raw_text))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    print("Bot iniciado!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
