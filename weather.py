import os
import time
import logging
import requests
from PIL import Image
import constants as con
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

PROCESSOR = None
MODEL = None

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")

def explain_img(path):

    processor = PROCESSOR
    model = MODEL
    model.to('cpu')
    
    raw_image = Image.open(path,"r").convert('RGB')

    # conditional image captioning
    prompt = "Question: In this picture, right now, is it raining? Answer:"

    inputs = processor(raw_image, text=prompt, return_tensors="pt").to('cpu')

    generated_ids = model.generate(**inputs, max_new_tokens=40)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print(prompt+generated_text)
    return prompt+" "+generated_text

async def rain(update: Update, context: ContextTypes.DEFAULT_TYPE):

    ppath = os.path.join(os.getcwd(),"current.png")

    if os.path.exists(ppath):
        os.system("rm {}".format(ppath))

    os.system('ffmpeg -i "$(yt-dlp -g SDQW6kWL1kY | head -n 1)" -vframes 1 {}'.format(ppath))

    time.sleep(5)

    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(ppath, 'rb'))
    await context.bot.send_message(chat_id=update.effective_chat.id, text=explain_img(ppath))

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text.replace("/echo ",""))
   

if __name__ == '__main__':

    PROCESSOR = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    MODEL = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

    application = ApplicationBuilder().token(con.TELEGRAM_KEY).build()
   
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    start_handler = CommandHandler('echo', echo)
    application.add_handler(start_handler)

    start_handler = CommandHandler('rain', rain)
    application.add_handler(start_handler)
   
    application.run_polling()