import os
import logging
from dotenv import load_dotenv
from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from ocr_processor import process_image
from mask_generator import (
    generate_mask,
    generate_raw_text_mask,
    list_available_masks,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

user_data: dict[int, dict] = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Bot de Mascara de Texto - Reagendamento 7017\n\n"
        "Envie um print/screenshot e depois use:\n"
        "/reagendamento - Gera a mascara preenchida\n"
        "/raw - Texto bruto do OCR"
    )


async def masks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(list_available_masks())


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

        response += "\nUse /reagendamento para gerar a mascara"
        await update.message.reply_text(response)

    except Exception as e:
        logger.error(f"Erro: {e}")
        await update.message.reply_text(f"Erro ao processar: {str(e)}")


async def mask_reagendamento(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id

    if user_id not in user_data:
        await update.message.reply_text("Envie um print primeiro.")
        return

    result = user_data[user_id]["result"]
    mask_text = generate_mask(result, "reagendamento")
    await update.message.reply_text(mask_text)


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
    await application.bot.set_my_commands([
        BotCommand("start", "Iniciar"),
        BotCommand("reagendamento", "Mascara reagendamento 7017"),
        BotCommand("raw", "Texto bruto OCR"),
    ])


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Defina TELEGRAM_BOT_TOKEN no .env")

    app = Application.builder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("masks", masks_cmd))
    app.add_handler(CommandHandler("reagendamento", mask_reagendamento))
    app.add_handler(CommandHandler("raw", raw_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    print("Bot iniciado!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
