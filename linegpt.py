import os
import io
import numpy as np
import openai
from PIL import Image
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, ImageMessage, TextSendMessage
from tensorflow.keras.models import load_model 
import logging
import firebase_admin
from firebase_admin import credentials,firestore
from google.cloud import firestore
from datetime import datetime,timedelta
import time 
import dotenv
from dotenv import load_dotenv


#試著用dotenv管理權限 

load_dotenv()

class tomatobot: 
    def __init__(self):
        #初始化linebot
        line_channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
        line_channel_secret = os.getenv('LINE_CHANNEL_SECRET')
        self.line_bot_api =LineBotApi(line_channel_access_token)
        self.handler = WebhookHandler(line_channel_secret)
        
        #設定openai API key 
        openai_api_key = os.getenv('OPENAI_API_KEY')
        #設置api密鑰
        open_ai_key = self.open_ai_key 

        # 設置 Google Cloud 認證
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"/Users/yonglinlai/Desktop/AItomatodetection system /tomatolinebot-fdadf536fe89.json"


        # 初始化 Firestore 客戶端
        self.db = firestore.Client(project=project_id, database=database_id)
        #指定專案資料庫
        project_id = "tomatolinebot"
        database_id = "default"
        print('Connection successful')
        logging.info('Firestore connection successful')
        # 加載模型 
        model_paths = {
            'ken':'/Users/yonglinlai/Desktop/tomatomodelv2/tomatovk8_trained_model.h5' #目前所訓練最高的val-accuracy0.82,但仍待提升
        }

        self.selected_model = 'ken'  
        self.model = load_model(model_paths[self.selected_model])

        
        #設置時間窗口
        self.TIME_WINDOW=timedelta(seconds=5)

        # 建立 Flask 應用程式
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route("/callback", methods=['POST'])
        def callback():
            signature = request.headers['X-Line-Signature']
            body = request.get_data(as_text=True)
            try:
                self.handler.handle(body, signature)
            except InvalidSignatureError:
                abort(400)
            return 'OK'
        
        self.handler.add(MessageEvent, message=TextMessage)(self.handle_text_message)
        self.handler.add(MessageEvent, message=ImageMessage)(self.handle_image_message)




    # 保存和查看 Firestore 中的對話歷史
    def save_chat_history(self,user_id, chat_history):
        doc_ref = self.db.collection('chat_histories').document(user_id)
        doc_ref.set({'history': chat_history, 'timestamp': datetime.now()})

    def get_chat_history(self,user_id):
        doc_ref = self.db.collection('chat_histories').document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return data.get('history', []), data.get('timestamp')
        else:
            return [], None

#先合併處理消息與圖片的功能  #待解決 如果同時傳圖片與文字,還是會分開回應,因為前置邏輯是圖片進入我的模型後分析結果出來再給ai生成,設置時間點
    def handle_combine_message(self,user_id, chat_history, disease_name=None):
        # 確定使用繁体中文
        system_message = {"role": "system", "content": "請使用繁體中文回答所有問題。"}

        # check是否已经有系统消息，如果没有則加
        if not any(msg["role"] == "system" for msg in chat_history):
            chat_history.insert(0, system_message)

        prompt = (
            f'你是一位專業植物識別分析專家,現在用戶有問題,請給出專業的建議'
        )
        if disease_name:
            prompt += f"模型分析结果是'{disease_name}'。請基於這個給予專業的防治建議,然後用繁體中文回答"

        chat_history.append({"role": "user", "content": prompt})

# 限制API的對話長度，保留最近的5條消息->加強之前回應要等老半天問題
        max_history_length = 5
        short_chat_history = chat_history[-max_history_length:]

        response = openai.ChatCompletion.create(
            model="gpt-4", #3.5更快
            messages=chat_history,
            max_tokens=1000,
            temperature=0.6,
        )
        final_response = response["choices"][0]["message"]["content"]

    # 回覆增加進去對話歷史
        chat_history.append({"role": "assistant", "content": final_response})
        self.save_chat_history(user_id, chat_history)
        # 回出結果給用戶
        reply = TextSendMessage(text=f"防治建議: {final_response}")
        return reply
    
# 當用戶給予文字的話直接讓gpt回覆
    def handle_text_message(self,event):
        user_id = event.source.user_id
        user_message = event.message.text

        # 如果講到分析直接跳過ai 省時間並引導用戶 ->讓使用者體驗變得順暢
        if "我想要分析" in user_message or "我想要分析圖片" in user_message or "請幫我分析照片" in user_message or "我想要分析照片" in user_message or "請幫我分析圖片" in user_message :
            reply = TextSendMessage(text="好的，請把圖片給我，讓我瞧瞧！")
            self.line_bot_api.reply_message(event.reply_token, reply)
            return  # 直接返回

     #獲取對話歷史和時間點
        chat_history, last_timestamp =self.get_chat_history(user_id)
        
        # 先判斷時間，是否是同一個消息用意
        if last_timestamp and datetime.now().replace(tzinfo=None) - last_timestamp.replace(tzinfo=None) <= TIME_WINDOW:
            # 判斷是同一組消息
            chat_history.append({"role": "user", "content": user_message})
            reply = self.handle_combine_message(user_id, chat_history)
        else:
            chat_history.append({"role": "user", "content": user_message})
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=chat_history,
                max_tokens=1000,
                temperature=0.5,
            )
            reply_text = response["choices"][0]["message"]["content"]
            chat_history.append({"role": "assistant", "content": reply_text})
            self.save_chat_history(user_id, chat_history)
            reply = TextSendMessage(text=reply_text)

    # 最終回應
        self.line_bot_api.reply_message(event.reply_token, reply)


# 當用戶給植物圖片,讓模型先分析,分析完的結果給到gpt 讓他生成回應 並給予建議
    
    def handle_image_message(self,event):
        try:
            user_id = event.source.user_id
            chat_history , last_timestamp = self.get_chat_history(user_id)

            # 等待2秒
            time.sleep(2)

            # 回覆用戶圖片正在分析中,作為立即訊息以此避免賴的reply token 過期
            self.line_bot_api.reply_message(event.reply_token, TextSendMessage(text="謝謝你的圖片提供,我正在透過模型分析中，請稍等一下..."))

            # 獲取圖片訊息
            message_content =self.line_bot_api.get_message_content(event.message.id)
            image = Image.open(io.BytesIO(message_content.content)).resize((224, 224)) #這裡要根據不同模型進行不同size的處理
            # 將圖片轉換為模型可接受的格式,讓模型可以去接受 /比例小於1所以255
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # 使用模型進行預測  / 自己的predicted 
            prediction = self.model.predict(image_array)
            predicted_class_index = np.argmax(prediction, axis=1)

            #加入我自己的類別
            disease_classes = ['powdery mildew', 'Tomato___Early_blight','Tomato___Late_blight','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus', 'Tomato___Bacterial_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Leaf_Mold','Gray spot','Tomato___healthy','Tomato___Septoria_leaf_spot','Tomato___Target_Spot','Other Disease']
            
            #確保在範圍內
            if predicted_class_index[0] < len(disease_classes):
                disease_name = disease_classes[predicted_class_index[0]]
            else:
                disease_name='Unknown Disease'

            # 如果是未知疾病，直接回覆，不用 OpenAI
            if disease_name == 'Unknown Disease':
                reply_text = "圖片有點不清楚，請幫忙再提供準確一點的圖片讓我幫你分析，謝謝。"
                self.line_bot_api.push_message(user_id, TextSendMessage(text=reply_text))
                return  # 直接返回
        
            # 等待2秒
            time.sleep(1)

        # 立即回覆用户模型分析结果
            initial_reply = TextSendMessage(text=f"這張圖片分析結果是'{disease_name}'請給我1分鐘,讓我給你專業的建議")
            self.line_bot_api.push_message(user_id, initial_reply)

    
            progress_reply = TextSendMessage(text="已經找到了正在生成建議中")
            self.line_bot_api.push_message(user_id, progress_reply)


        # 使用OpenAI生成自然語言回覆，附上建議
            reply =self.handle_combine_message(user_id, chat_history, disease_name)

        # 回覆給用戶
            self.line_bot_api.push_message(user_id, reply)

        except Exception as e :
            logging.error(f"Error in processing image message: {e}")
            self.line_bot_api.push_message(user_id, TextSendMessage(text="在處理圖片時出現錯誤，請稍後再試。"))
       
    def run(self):
        port = int(os.environ.get('PORT',8500))
        self.app.run(host='0.0.0.0',port=port, debug=True)


if __name__ == "__main__":
    bot =tomatobot()
    bot.run()


