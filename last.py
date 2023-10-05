import pymysql
import os
import json
import time
import gc
import torch
from tqdm import tqdm
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from jinja2 import Environment, FileSystemLoader
from flask import Flask, render_template, request, redirect, session
from main_copy import storyfile,text_to_image,text_to_speech
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
org_path=os.getcwd()
from experiment import TTS_train
from Tacotron2_inference import my_tts
import base64
import numpy as np

#summarization
checkpoint_summarization = org_path+"/t5-large-korean-text-summary"
summarizer=pipeline('summarization',checkpoint_summarization,max_length=40)
#translation
checkpoint_translation = org_path+"/translation"
translator = pipeline("translation", checkpoint_translation,max_length=256)
#text-to-image
model_base = org_path+"/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)





app = Flask(__name__)
app.secret_key = 'your-secret-key' 

# MySQL 연결 설정
mysql = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='1234',
    charset='utf8mb4'
)
cursor = mysql.cursor()

# 데이터베이스 조회 쿼리 실행
cursor.execute("SHOW DATABASES;")
databases = cursor.fetchall()

# 데이터베이스 출력
db_list = []
for db in databases:
    db_list.append(db[0])

if 'readme' not in db_list:
    sql = 'CREATE DATABASE readme;'
    cursor.execute(sql)

cursor.execute('USE readme;')

cursor.execute("SHOW TABLES;")
tables = cursor.fetchall()

table_list = []
for table in tables:
    table_list.append(table[0])

if 'users' not in table_list:
    sql = """CREATE TABLE users (
                id INT NOT NULL AUTO_INCREMENT,
                username VARCHAR(50) NOT NULL,
                password VARCHAR(50) NOT NULL,
                PRIMARY KEY (id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"""
    cursor.execute(sql)


if 'static' not in os.listdir():
    os.mkdir('static')
if 'images' not in os.listdir('static'):
    os.mkdir(org_path+'/static/images')
if 'fiction' not in os.listdir('static'):
    os.mkdir(org_path+'/static/fiction')
if 'voice_actor' not in os.listdir('static'):
    os.mkdir(org_path+'/static/voice_actor')
if 'result_folder' not in os.listdir('static'):
    os.mkdir(org_path+'/static/result_folder')
if 'wave_zip' not in os.listdir('static'):
    os.mkdir(org_path+'/static/wave_zip')    
if 'wave_txt' not in os.listdir('static'):
    os.mkdir(org_path+'/static/wave_txt')  
if 'json' not in os.listdir('static'):
    os.mkdir(org_path+'/static/json')
if 'css' not in os.listdir('static'):
    os.mkdir(org_path+'/static/css')
if 'readme.json' not in os.listdir(org_path+'/static/json'):
    with open(org_path+'/static/json/readme.json','w') as json_file:
        data={}
        json.dump(data,json_file)

# 회원가입
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # 중복 확인
        select_query = "SELECT * FROM users WHERE username = %s"
        select_data = (username,)
        cursor.execute(select_query, select_data)
        result = cursor.fetchone()

        if result:
            message = '이미 있는 ID입니다.'
            return render_template('register.html', message=message)
        else:
            insert_query = "INSERT INTO users (username, password) VALUES (%s, %s)"
            insert_data = (username, password)
            cursor.execute(insert_query, insert_data)
            mysql.commit()
            return redirect('/login')
    else:
        return render_template('register.html')

# 로그인
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        select_query = "SELECT * FROM users WHERE username = %s AND password = %s"
        select_data = (username, password)
        cursor.execute(select_query, select_data)
        result = cursor.fetchone()

        if result:
            session['username'] = username  # 세션에 username 저장
            return redirect('/'+username)
        else:
            message = '잘못된 계정입니다.'
            return render_template('login.html', message=message)
    else:
        return render_template('login.html')

@app.route('/check_username')
def check_username():
    username = request.args.get('username')

    select_query = "SELECT * FROM users WHERE username = %s"
    select_data = (username,)
    cursor.execute(select_query, select_data)
    result = cursor.fetchone()

    if result:
        message = '이미 있는 아이디입니다.'
    else:
        message = '생성 가능합니다.'

    return render_template('register.html', message=message)

# 메인 페이지
@app.route('/<username>')
def main(username):

    with open(org_path+'/static/json/readme.json','r',encoding='UTF-8') as json_file:
        data=json.load(json_file)

        if username not in data:
            data[username]={}
            data[username]['fiction_list']={}
            data[username]['zip_files']=[]
            data[username]['txt_files']=[]
            data[username]['voice_actor']={}
            with open(org_path+'/static/json/readme.json','w') as json_file:
                json.dump(data,json_file)
    fiction_list=data[username]['fiction_list']

    
    return render_template('main.html', username=username,fiction_list=fiction_list)

@app.route('/<username>/management', methods=['GET','POST'])
def management(username):
    with open(org_path+'/static/json/readme.json','r',encoding='UTF-8') as json_file:
        data=json.load(json_file)
    zip_files = data[username]['zip_files']
    txt_files = data[username]['txt_files']
    voice_actor=data[username]['voice_actor']
    fiction_list=data[username]['fiction_list']
    return render_template('management.html', username=username,fiction_list=fiction_list,zip_files=zip_files,txt_files=txt_files,voice_actor=voice_actor)

def get_zip_files():
    files = os.listdir(org_path+'/static/wave_zip')
    zip_files = [file for file in files if file.endswith('.zip')]
    return zip_files
def get_txt_files():
    files = os.listdir(org_path+'/static/wave_txt')
    txt_files = [file for file in files if file.endswith('.txt')]
    return txt_files





@app.route('/<username>/record', methods=['GET','POST'])
def record(username):
    
    return render_template('record.html', username=username)

@app.route('/<username>/upload', methods=['GET','POST'])
def upload(username):
    if request.method=='POST':
        f=request.files['file']
        if '.txt'==f.filename[-4:]:
            f.save(org_path+'/static/fiction/'+f.filename)
            full_text,Page_text=storyfile(org_path+'/static/fiction/'+f.filename)
            with open(org_path+'/static/json/readme.json','r',encoding='UTF-8') as json_file:
                data=json.load(json_file)
                data[username]['fiction_list'][f.filename[:-4]]={}
            with open(org_path+'/static/json/readme.json','w') as json_file:
                json.dump(data,json_file)
            with open(org_path+'/static/json/readme.json','r',encoding='UTF-8') as json_file:
                data=json.load(json_file)
                for idx,sentence in enumerate(Page_text):
                    data[username]['fiction_list'][f.filename[:-4]][idx]={}
                    data[username]['fiction_list'][f.filename[:-4]][idx]['choice']=[]
                    data[username]['fiction_list'][f.filename[:-4]][idx]['image_path']=[]
                    data[username]['fiction_list'][f.filename[:-4]][idx]['voice_type']=[]
                    data[username]['fiction_list'][f.filename[:-4]][idx]['full_text']=[sentence]
                    data[username]['fiction_list'][f.filename[:-4]][idx]['translation+summary_text']=[translator(summarizer(sentence)[0]['summary_text'])[0]['translation_text'].replace('.',' ').strip()+'.']           
            with open(org_path+'/static/json/readme.json','w') as json_file:
                json.dump(data,json_file)
        if '.' not in f.filename:
            f.save(org_path+'/static/voice_actor/'+username+'_'+f.filename)
        if '.zip'==f.filename[-4:]:
            f.save(org_path+'/static/wave_zip/'+f.filename)
        if (f.filename[-4:]=='.txt') and (f.filename[:-4] not in os.listdir(org_path+'/static/images')):
            os.mkdir(org_path+'/static/images/'+f.filename[:-4])
        return redirect('/'+username+'/management')
    else:
        return render_template('management.html', username=username)



@app.route('/<username>/upload_train', methods=['GET','POST'])
def upload_train(username):
    if request.method=='POST':
        f=request.files['file']
        with open(org_path+'/static/json/readme.json','r',encoding='UTF-8') as json_file:
            data=json.load(json_file)
        if '.txt'==f.filename[-4:]:
            f.save(org_path+'/static/wave_txt/'+f.filename)
            data[username]['txt_files'].append(f.filename)
        elif '.zip'==f.filename[-4:]:
            f.save(org_path+'/static/wave_zip/'+f.filename)
            data[username]['zip_files'].append(f.filename)    
        elif '.' not in f.filename:
            f.save(org_path+'/static/voice_actor/'+username+'_'+f.filename)   
        with open(org_path+'/static/json/readme.json','w') as json_file:
            json.dump(data,json_file)
         
        return redirect('/'+username+'/management')
    else:
        return render_template('management.html', username=username)    



@app.route('/<username>/train', methods=['GET','POST'])
def training(username):
    if request.method=='POST':
        wav_file = request.form['wav_file']
        wav_text = request.form['wav_text']
        model_name = request.form['model_name']
        TTS_train(org_path=org_path,wav_zip=wav_file,wav_txt=wav_text,custom_model_name=model_name)

        return redirect('/'+username+'/management')
    else:
        return render_template('management.html', username=username)





@app.route('/<username>/<fiction>', methods=['GET','POST'])
def fiction(username, fiction):
    

    with open(org_path+'/static/json/readme.json', 'r', encoding='UTF-8') as json_file:
        data = json.load(json_file)
        image_path = []
        full_text = []
        for page in data[username]['fiction_list'][fiction]:
            image_path.extend(data[username]['fiction_list'][fiction][page]['choice'])
            full_text.append(data[username]['fiction_list'][fiction][page]['full_text'][0])

    return render_template('fiction.html', username=username, fiction=fiction, image_path=image_path, full_text=full_text)

@app.template_filter('b64encode')
def b64encode_filter(data):
    return base64.b64encode(data).decode('utf-8')

@app.route('/<username>/<fiction>/<page_number>', methods=['GET','POST'])
def fiction_detail(username,fiction,page_number):

    wav_file_list = ['result_folder/sounds.wav']
    
   
    with open(org_path+'/static/json/readme.json','r',encoding='UTF-8') as json_file:
        data=json.load(json_file)
        full_text=[data[username]['fiction_list'][fiction][page_number]['full_text'][0]]
        choice_image=data[username]['fiction_list'][fiction][page_number]['choice']
    
    return render_template('fiction_detail.html', username=username,fiction=fiction,full_text=full_text,page_number=page_number,choice_image=choice_image,wav_file_list=wav_file_list)




@app.route('/<username>/<fiction>/management', methods=['GET','POST'])
def management_fiction(username,fiction):
    with open(org_path+'/static/json/readme.json','r',encoding='UTF-8') as json_file:
        data=json.load(json_file)
        if username not in data:
            data[username]={}
            with open(org_path+'/static/json/readme.json','w') as json_file:
                json.dump(data,json_file)
        fiction_list=data[username]['fiction_list'].keys() 
        image_path=[]
        full_text=[]
        for page in data[username]['fiction_list'][fiction]:
            image_path.extend(data[username]['fiction_list'][fiction][page]['image_path'])
            full_text.append(data[username]['fiction_list'][fiction][page]['full_text'][0])
        
       
    return render_template('management_fiction.html', username=username,fiction=fiction,fiction_list=fiction_list,image_path=image_path,full_text=full_text)


@app.route('/<username>/<fiction>/<page_number>/management', methods=['GET','POST'])
def management_fiction_detail(username,fiction,page_number):
    if request.method == 'POST':        
        # Get data from the HTML form
        with open(org_path+'/static/json/readme.json','r',encoding='UTF-8') as json_file:
            data=json.load(json_file)
        form_data=data[username]['fiction_list'][fiction][page_number]['translation+summary_text'][0]
        # Process the form data or pass it to the model   
        if fiction=='빨간모자':
            pipe.unet.load_attn_procs(org_path+'/dreambooth-Lora_redhat')
            if 'red hat' in form_data:
                form_data.replace('red hat','sks red hat')
        pipe.to("cuda")
        image = pipe("high quality a photo of %s"%form_data, num_inference_steps=100, guidance_scale=15).images[0]

        time_str=time.strftime('%Y-%m-%d-%H-%M-%S')
        data[username]['fiction_list'][fiction][page_number]['image_path'].append("%s.png"%time_str)
        with open(org_path+'/static/json/readme.json','w') as json_file:
            json.dump(data, json_file)
        image.save(org_path+"/static/images/"+fiction+"/%s.png"%time_str)   
    with open(org_path+'/static/json/readme.json','r',encoding='UTF-8') as json_file:
        data=json.load(json_file)
        image_path=data[username]['fiction_list'][fiction][page_number]['image_path']
        full_text=[data[username]['fiction_list'][fiction][page_number]['full_text'][0]]
        fiction_list=data[username]['fiction_list'].keys()
    wav_file_list = ['result_folder/sounds.wav']
    return render_template('management_fiction_detail.html', username=username,fiction=fiction,page_number=page_number,fiction_list=fiction_list,image_path=image_path,full_text=full_text,wav_file_list=wav_file_list)

@app.route('/<username>/<fiction>/<page_number>/management/choice', methods=['GET','POST'])
def management_fiction_detail_choice(username,fiction,page_number):
    if request.method == 'POST':
        image_url = request.form.get('image_url')
        with open(org_path+'/static/json/readme.json','r',encoding='UTF-8') as json_file:
            data=json.load(json_file)   
        if not data[username]['fiction_list'][fiction][page_number]['choice']:
            data[username]['fiction_list'][fiction][page_number]['choice'].append(image_url.split('/')[-1])
        else:
            data[username]['fiction_list'][fiction][page_number]['choice']=[image_url.split('/')[-1]]
        with open(org_path+'/static/json/readme.json','w') as json_file:
            json.dump(data, json_file)
        return redirect('/'+username+'/'+fiction+'/'+page_number+'/management')
    else:
        with open(org_path+'/static/json/readme.json','r',encoding='UTF-8') as json_file:
            data=json.load(json_file)
            image_path=data[username]['fiction_list'][fiction][page_number]['image_path']
            full_text=[data[username]['fiction_list'][fiction][page_number]['full_text'][0]]
            fiction_list=data[username]['fiction_list'].keys()
        return render_template('management_fiction_detail.html', username=username,fiction=fiction,page_number=page_number,fiction_list=fiction_list,image_path=image_path,full_text=full_text)

@app.route('/<username>/<fiction>/<page_number>/management/delete', methods=['GET','POST'])
def management_fiction_detail_delete(username,fiction,page_number):
    if request.method == 'POST':
        image_url = request.form.get('image_url')
        with open(org_path+'/static/json/readme.json','r',encoding='UTF-8') as json_file:
            data=json.load(json_file)  
        if not data[username]['fiction_list'][fiction][page_number]['choice']:
            pass
        else:
            if image_url.split('/')[-1] in data[username]['fiction_list'][fiction][page_number]['choice']:
                data[username]['fiction_list'][fiction][page_number]['choice']=[]
            else:
                pass
        data[username]['fiction_list'][fiction][page_number]['image_path'].remove(image_url.split('/')[-1])
        os.remove(image_url[1:])
        with open(org_path+'/static/json/readme.json','w') as json_file:
            json.dump(data, json_file)
        return redirect('/'+username+'/'+fiction+'/'+page_number+'/management')
    else:   
        with open(org_path+'/static/json/readme.json','r',encoding='UTF-8') as json_file:
            data=json.load(json_file)
            image_path=data[username]['fiction_list'][fiction][page_number]['image_path']
            full_text=[data[username]['fiction_list'][fiction][page_number]['full_text'][0]]
            fiction_list=data[username]['fiction_list'].keys()
        return render_template('management_fiction_detail.html', username=username,fiction=fiction,page_number=page_number,fiction_list=fiction_list,image_path=image_path,full_text=full_text)
    


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

# 연결 종료
cursor.close()
mysql.close()