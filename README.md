# Bot Telegram - Mascara de Texto OCR

Bot que recebe prints/screenshots, extrai texto via OCR e gera mascaras de texto preenchidas.

## Requisitos

- Python 3.10+
- Tesseract OCR instalado no sistema

## Instalacao do Tesseract OCR

### Windows
1. Baixe em: https://github.com/UB-Mannheim/tesseract/wiki
2. Instale e adicione ao PATH
3. Para suporte a portugues, baixe os traineddata em:
   https://github.com/tesseract-ocr/tessdata
   Coloque o arquivo `por.traineddata` na pasta `tessdata` do Tesseract

### Linux (Ubuntu/Debian)
```bash
sudo apt install tesseract-ocr tesseract-ocr-por
```

### macOS
```bash
brew install tesseract tesseract-lang
```

## Configuracao

1. Crie um bot no Telegram com @BotFather e copie o token
2. Crie o arquivo `.env` a partir do exemplo:
   ```bash
   cp .env.example .env
   ```
3. Edite `.env` e cole o token:
   ```
   TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
   ```

## Instalacao das dependencias

```bash
pip install -r requirements.txt
```

## Execucao

```bash
python bot.py
```

## Uso

1. Envie `/start` para o bot
2. Envie uma imagem/print/screenshot
3. O bot extrai o texto e identifica campos automaticamente
4. Escolha uma mascara:
   - `/mask_simples` - Lista todos os campos encontrados
   - `/mask_cadastro` - Formulario de cadastro
   - `/mask_financeiro` - Extrato financeiro
   - `/mask_contato` - Ficha de contato
   - `/raw` - Texto bruto do OCR

## Campos detectados automaticamente

- CPF, CNPJ, RG
- Datas (DD/MM/AAAA)
- Horas (HH:MM)
- Emails
- Telefones
- CEPS
- Moedas (R$)
- Numeros
- Nomes proprios
- IPs e URLs
- Campos no formato `label: valor`

## Estrutura

```
telegram-text-mask-bot/
  bot.py              # Bot principal do Telegram
  ocr_processor.py    # Modulo de extracao de texto (OCR)
  mask_generator.py   # Gerador de mascaras
  requirements.txt    # Dependencias Python
  .env.example        # Exemplo de configuracao
```
