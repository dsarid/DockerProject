import flask
from flask import request
import os
from bot import Bot, QuoteBot, ObjectDetectionBot

app = flask.Flask(__name__)


token_secret_file = open('/run/secrets/telegram_bot_token.secret', 'r')
TELEGRAM_TOKEN = token_secret_file.read().rstrip()
token_secret_file.close()

TELEGRAM_APP_URL = os.environ['TELEGRAM_APP_URL']
images_bucket = os.environ['BUCKET_NAME']


@app.route('/', methods=['GET'])
def index():
    return 'Ok'


@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'


if __name__ == "__main__":
    bot = ObjectDetectionBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL, images_bucket)

    app.run(host='0.0.0.0', port=8443)
