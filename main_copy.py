import os
import time
from tqdm import tqdm
import gc
from transformers import pipeline
import torch
from diffusers import StableDiffusionPipeline
from Tacotron2_inference import my_tts
import experiment
import json

# 소설 txt input(ex 빨간모자)
def storyfile(textfile):

    fiction=textfile #'fiction/빨간모자.txt'
    with open(fiction,'r',encoding='UTF-8') as f:
        k=f.readlines()
        sentence=''
        paragraph=[]
    for idx,text in enumerate(k):
        if text=='\n':
            continue
        else:
            paragraph.append(text)
    full_story=[]
    for paragraph_unit in tqdm(paragraph):
        sentence=''
        for idx,text in enumerate(paragraph_unit):
            sentence+=text
            if (text=='.') or (text=='!') or (text=='?'):
                if (len(sentence)>=200) and (sentence.count('\'')%2==0) and (sentence.count('\"')%2==0): 
                    full_story.append(sentence)
                    sentence=''
            if idx==len(paragraph_unit)-1:
                if len(sentence)<=100:
                    if full_story:
                        full_story[-1]=full_story[-1]+sentence
                    else:
                        full_story.append(sentence)
                else:
                    full_story.append(sentence)
    return paragraph,full_story


def text_to_image(full_story,file_name):
    
    
    
    checkpoint_summarization = "C:/real_alpaco/t5-large-korean-text-summary"
    checkpoint_translation = "C:/real_alpaco/translation"
    summarizer=pipeline('summarization',checkpoint_summarization,max_length=40)
    translator = pipeline("translation", model=checkpoint_translation,max_length=256)

    start_time=time.time()
    preprocess_sentence=[]
    for snt in tqdm(full_story):
        # print("sentence: ",snt)
        # print("summarizer: ", summarizer(snt)[0]['summary_text'])    
        #preprocess_sentence.append(summarizer(snt)[0]['summary_text'])
        preprocess_sentence.append(translator(summarizer(snt)[0]['summary_text'])[0]['translation_text'])
    with open('static/json/alpaco.json','r',encoding='UTF-8') as json_file:
        data=json.load(json_file)
        for su in range(len(full_story)):
            if su not in  data['fiction_list'][file_name]:
                data['fiction_list'][file_name][su]={}
                data['fiction_list'][file_name][su]['choice']=[]
                data['fiction_list'][file_name][su]['image_path']=[]
                data['fiction_list'][file_name][su]['voice_type']={}
                data['fiction_list'][file_name][su]['text']=preprocess_sentence[su]
    with open('static/json/alpaco.json','w') as json_file:
        json.dump(data, json_file)
    end_time=time.time()-start_time
    print('입력 프롬포트를 작성중입니다 준비 시간: ',end_time)



    #text to image:

    model_base = "C:/real_alpaco/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
    #예시 빨간모자
    pipe.unet.load_attn_procs('C:/real_alpaco/dreambooth-Lora_redhat')
    pipe.to("cuda")
    print('텍스트를 이미지로 변환중입니다...')

    if 'images' not in os.listdir():
        os.mkdir('images')
    for su,split_text in enumerate(preprocess_sentence):
        #print(split_text)
        start_time=time.time()
        if 'red hat' in split_text:
            split_text.replace('red hat','sks red hat')
        image = pipe("a photo of %s"%split_text, num_inference_steps=25, guidance_scale=7.5).images[0]

        time_str=time.strftime('%Y-%m-%d-%H-%M-%S')
        with open('static/json/alpaco.json','r',encoding='UTF-8') as json_file:
            data=json.load(json_file)
            data['fiction_list'][file_name][str(su)]['image_path'].append("%s.png"%time_str)
        with open('static/json/alpaco.json','w') as json_file:
            json.dump(data, json_file)
        image.save("static/images/"+file_name+"/%s.png"%time_str)
        
        end_time=time.time()-start_time
        print('이미지 생성 시간:',end_time)
        gc.collect()
        torch.cuda.empty_cache()

    #import Tacotron2

    
    
def text_to_speech(full_story):
    while(1):
        voice=input('음성합성을 시작합니다. 다른 목소리로 하고 싶습니까? Y/N: ')
        if voice=='Y' or voice=='y':
            print('다른 목소리로 하겠습니다.')
            pwd=os.getcwd()
            wav_zip='마루halflife.zip'
            wav_txt='halflife.txt'
            new_name='알파코5기'
            experiment.TTS_train(org_path=pwd,wav_zip=wav_zip,wav_txt=wav_txt,custom_model_name=new_name)
            break
        elif voice=='N' or voice=='n':
            print('디폴트로 하겠습니다.')
            my_tts(full_story,'진성아')
            break
        else:
            print('다시 입력해주십시오. Y/N')



    # if 'fiction' not in os.listdir():
    #     os.mkdir('fiction')


    # os.listdir
    # with open('fiction')

