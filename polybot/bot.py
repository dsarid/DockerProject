import telebot
from loguru import logger
import os
import time
from datetime import datetime
from telebot.types import InputFile
from img_proc import Img
import boto3
import polybot_helper_lib
import requests


class Bot:

    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class QuoteBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if msg["text"] != 'Please don\'t quote me':
            self.send_text_with_quote(msg['chat']['id'], msg["text"], quoted_msg_id=msg["message_id"])


class ObjectDetectionBot(Bot):
    def __init__(self, token, telegram_chat_url, images_bucket):
        super().__init__(token, telegram_chat_url)
        self.media_group = None
        self.filter = None
        self.filters_list = ["Blur", "Contour", "Rotate", "Segment", "Salt and pepper", "Concat", "Segment"]
        self.previous_pic = None
        self.images_bucket = images_bucket


    @staticmethod
    def _apply_filter(img, filter):
        if filter == "Blur":
            img.blur()
        elif filter == "Contour":
            img.contour()
        elif filter == "Rotate":
            img.rotate()
        elif filter == "Segment":
            img.segment()
        elif filter == "Salt and pepper":
            img.salt_n_pepper()


    def add_date_to_filename(self, file_path):
        # Split the file path into directory and filename
        directory, filename = os.path.split(file_path)

        # Get the current date
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Extract file extension
        name, extension = os.path.splitext(filename)

        # Create the new filename with the date appended
        new_filename = f"{name}_{current_date}{extension}"

        # Construct the new file path
        new_file_path = os.path.join(directory, new_filename)

        try:
            # Rename the file
            os.rename(file_path, new_file_path)
            print(f"File renamed to: {new_filename}")
            return new_file_path
        except Exception as e:
            print(f"Error: {e}")
        return None


    def handle_message(self, msg):

        # TODO upload the photo to S3
        # TODO send an HTTP request to the `yolo5` service for prediction
        # TODO send the returned results to the Telegram end-user

        logger.info(f'Incoming message: {msg}')
        # check if the message contain a media group, this is happening when the user send more than one images at once
        if self.media_group is None:
            self.media_group = msg.get("media_group_id")
        elif self.media_group != msg.get("media_group_id"):
            self.media_group = msg.get("media_group_id")
            # resetting the filter in case the user send another group of pictures
            self.filter = None

        if self.filter is None or msg.get("media_group_id") is None:
            # when getting an image without a media_group_id, we put the "caption" value in .filter
            if msg.get("caption"):
                self.filter = msg.get("caption")
            else:
                self.send_text(
                    msg['chat']['id'],
                    f"You have to provide a picture and one of the following filters: {self.filters_list}"
                )
                return None

        if self.filter == "Concat":
            if msg.get("media_group_id"):
                if msg.get("caption"):
                    photo_path = self.download_user_photo(msg)
                    process_photo = Img(photo_path)
                    self.previous_pic = process_photo
                else:
                    # this is a special case for using concat,
                    # the concat caption is excepted to be sent with two pictures, the first one with caption...
                    photo_path = self.download_user_photo(msg)
                    process_photo = Img(photo_path)
                    process_photo.concat(self.previous_pic)
                    processed_pic = process_photo.save_img()
                    self.send_photo(msg['chat']['id'], processed_pic)

        elif msg.get("media_group_id") is None:
            if self.filter in self.filters_list:
                try:
                    photo_path = self.download_user_photo(msg)
                    process_photo = Img(photo_path)
                    self._apply_filter(process_photo, self.filter)
                    self.filter = None
                    processed_pic = process_photo.save_img()
                    self.send_photo(msg['chat']['id'], processed_pic)
                except Exception as E:
                    self.send_text(
                        msg['chat']['id'],
                        f"An error occurred. You have to provide a picture and one of the following filters: {self.filters_list}"
                    )
                    logger.info(f"ERR: {E}")
            elif self.filter == "Predict":

                images_dir = "photos/predicted_images"

                photo_path = self.download_user_photo(msg)
                photo_path = self.add_date_to_filename(photo_path)
                s3 = boto3.client('s3')
                polybot_helper_lib.upload_file(photo_path, self.images_bucket, s3)
                yolo5_base_url = "http://yolo5:8081/predict"
                s3_img_name = os.path.split(photo_path)
                yolo5_url = f"{yolo5_base_url}?imgName={s3_img_name[1]}"
                for i in range(5):
                    try:
                        logger.info(f"File name: {s3_img_name[1]}")
                        s3.head_object(
                            Bucket=self.images_bucket,
                            Key=s3_img_name[1]
                        )
                        break
                    except Exception as E:
                        logger.info(f"file probably not there yet :/ attempt no: {i}")
                        time.sleep(5)
                try:
                    response = requests.post(yolo5_url)
                    if not os.path.exists(images_dir):
                        os.makedirs(images_dir)

                    s3.download_file(os.environ["BUCKET_NAME"], f"predicted_img/{s3_img_name[1]}", f'{images_dir}/{s3_img_name[1]}')

                    results_json = response.json().get('labels')

                    results = polybot_helper_lib.count_objects_in_dict(results_json)

                    logger.info(f"results are: {results}")

                    self.send_photo(msg['chat']['id'], f'{images_dir}/{s3_img_name[1]}')
                    self.send_text(msg['chat']['id'], str(results))
                    print(response.status_code)
                except Exception as e:
                    print("WHY???")
                    print(e)
                print(f"processing: {yolo5_url}")

                self.send_text(
                    msg['chat']['id'],
                    f"Processing: {yolo5_url}"
                )


            else:
                self.send_text(
                    msg['chat']['id'],
                    f"An error occurred. You have to provide a picture and one of the following filters: {self.filters_list}"
                )
                self.filter = None

        else:
            if self.filter in self.filters_list:
                try:
                    photo_path = self.download_user_photo(msg)
                    process_photo = Img(photo_path)
                    self._apply_filter(process_photo, self.filter)
                    processed_pic = process_photo.save_img()
                    self.send_photo(msg['chat']['id'], processed_pic)
                except Exception:
                    self.send_text(
                        msg['chat']['id'],
                        f"An error occurred. ou have to provide a picture and one of the following filters: {self.filters_list}"
                    )
            else:
                self.send_text(
                    msg['chat']['id'],
                    f"An error occurred. ou have to provide a picture and one of the following filters: {self.filters_list}"
                )
                self.filter = None
