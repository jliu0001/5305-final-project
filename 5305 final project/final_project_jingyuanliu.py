import websocket
import speech_recognition as sr
import threading
import uuid
import json
from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode
import ssl
import requests
from datetime import datetime

"""
1. Connect `ws_app.run_forever()` # Main function
2. After a successful connection, send data `on_open()`
2.1 Send the start parameters frame `send_start_params()`
2.2 Send the audio data frame `send_audio()`
2.3 The library receives the recognition result `on_message()`
2.4 Send the end frame `send_finish()`
3. Close the voice recognition connection `on_close()`
4. Get the token for emotion recognition `get_token()`
5. Call the emotion recognition API `get_label_baidu()`
"""

def send_start_params(ws):
    """
    :param websocket.WebSocket ws:
    :return:
    """
    req = {
        "type": "START",
        "data": {
            "appid": Voc_2_text_APPID,  # appid
            "appkey": Voc_2_text_APPKEY,  # appkey
            "dev_pid": DEV_PID,  # model
            "cuid": "1234",  # any number
            "sample": 16000,  # paramater
            "format": "pcm"  #paramater
        }
    }
    body = json.dumps(req)
    ws.send(body, websocket.ABNF.OPCODE_TEXT)

def send_audio(ws):
    try:
        r = sr.Recognizer()
        mic = sr.Microphone()
        print("***************Recording*****************")
        with mic as source:
            r.adjust_for_ambient_noise(source)
            print("Please Speak...")
            audio = r.listen(source)
        print("End Recording,Sending Message...")
        body = audio.get_wav_data(convert_rate=16000)
        ws.send(body, websocket.ABNF.OPCODE_BINARY)
        print("Done Sending")
    except Exception as e:
        print(f"Wrong: {e}")

def send_finish(ws):
    """
    :param websocket.WebSocket ws:
    :return:
    """
    req = {"type": "FINISH"}
    body = json.dumps(req)
    ws.send(body, websocket.ABNF.OPCODE_TEXT)

def on_open(ws):
    """
    :param  websocket.WebSocket ws:
    :return:
    """
    def run(*args):
        send_start_params(ws)
        send_audio(ws)
        send_finish(ws)
    threading.Thread(target=run).start()

def on_message(ws, message):
    try:
        message = json.loads(message)
        print(message)
        text = message.get("result", "")
        if message["type"] == "FIN_TEXT" and text:
            print(f"Recognize Text: {text}")
            token = get_token()
            if token:
                get_label_baidu(text, token)
            else:
                print("fail to get token!")
    except Exception as e:
        print(f"wrong: {e}")


def get_token():
    ssl._create_default_https_context = ssl._create_unverified_context
    # OCR_URL = "https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify"
    TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'
    params = {'grant_type': 'client_credentials',
              'client_id': E_analy_API_KEY,
              'client_secret': E_analy_SECRET_KEY}
    post_data = urlencode(params)
    result_str = ''
    post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req, timeout=5)
        result_str = f.read()
    except URLError as err:
        print(err)
    result_str = result_str.decode()
    result = json.loads(result_str)

    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if not 'brain_all_scope' in result['scope'].split(' '):
            print('please ensure has check the  ability')
            exit()
        token = result['access_token']
        return token
    return None

def get_label_baidu(text,token):
    # Save text data in variable new_each
    new_each = {'text': text }
    new_each = json.dumps(new_each)

    url = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token={}'.format(token)
    res = requests.post(url, data=new_each)  # Use URL to request Baidu Sentiment Analysis API
    if res.status_code == 200:
        res_text = res.text
        result = res_text.find('items')
        if result == -1:
            print("wrong")
            return 0
        else:
            json_data = json.loads(res.text)
            value = (json_data['items'][0]['positive_prob'])  # Get sentiment index value
            if value > 0.5:
                print("Positive" , value)
            else:
                print("Negative",value)
    else:
        print("wrong")
        return 0

if __name__ == "__main__":
    Voc_2_text_APPID = 42659263  #  APPID
    Voc_2_text_APPKEY = "nnuDDWUNvqC8tHMfnnSbDSuQ" #  APPKEY
    E_analy_API_KEY = "nnuDDWUNvqC8tHMfnnSbDSuQ"   # APIKEY
    E_analy_SECRET_KEY = "Lsbpfj4vAFmCG2CKyH0W4rdAU8ruYf2C" # Secret key
    #  Language model, which can be modified for other language model tests, such as far-field Mandarin 19362
    DEV_PID = 15372
    URI = "ws://vop.baidu.com/realtime_asr"

    uri = URI + "?sn=" + str(uuid.uuid1())
    def on_error(ws, error):
        print("ERROR:", error)

    def on_close(ws):
       print("### closed ###")

    ws_app = websocket.WebSocketApp(uri,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)  
    ws_app.run_forever()
