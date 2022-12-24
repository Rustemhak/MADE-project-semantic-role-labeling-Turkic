from django.http import HttpResponse
from django.shortcuts import render
from .forms import InputForm

import allennlp_models.tagging
import numpy as np
import pandas as pd
import torch

from allennlp.predictors.predictor import Predictor
from bertviz import head_view, model_view
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import requests
import json
import time

def translate(texts, source_language='tt', target_language='en'):
    IAM_TOKEN = 't1.9euelZrOkM2UksuclovNjcmKzJiZx-3rnpWaj46Nkc6ejZaLyZvOmZqWj53l8_dyf29i-e8BUzNq_N3z9zIubWL57wFTM2r8.uXM6svBjxv9pr7FZya6q_Ht4mX8lukqtZjKcnZlckwZ6T285-HfQG9jEf1Kas54sZx2-5YeeiO5HBuhQzKkhBA'
    folder_id = 'b1grvnc3e8sgtm7qcsja'
    body = {
        "targetLanguageCode": target_language,
        "texts": texts,
        "folderId": folder_id,
        "sourceLanguageCode":  source_language
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {0}".format(IAM_TOKEN)
    }
    url = 'https://translate.api.cloud.yandex.net/translate/v2/translate'
    response = requests.post(url,
        json = body,
        headers = headers
    )
    if response.status_code != 200:
        print('Ожидаю 0.5 секунды...')  
        time.sleep(0.5)
        response = requests.post(url,
        json = body,
        headers = headers
        )
    d = json.loads(response.text)
    translations = d['translations']
    return [t['text'] for t in translations] 


f = open("/home/vs/roberta/config.json", "r")
print(f.read()) 

class Model_srl:
    def __init__(self, path_to_model = "/home/vs/roberta"):
        self.model = AutoModelForSequenceClassification.from_pretrained(path_to_model, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.sep_token = " </s> "
        self.srl_model = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
        
        self.att_threshold = 0.1e-00
        self.att_n_tok_per_word = 1
        self.att_layer = 6
        self.att_head = 2
        
    def get_vocab(self, sentence):
        self.tokenized = self.tokenizer([sentence], return_tensors='pt').to(self.model.device)
        self.tokens = self.tokenizer.convert_ids_to_tokens(self.tokenized['input_ids'][0].tolist())
        self.tok_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_tok = {idx: token  for idx, token in enumerate(self.tokens)}
    
    def get_word_to_tok(self, desired_output, sentence):
        word2tok_dict = {x:desired_output[i] for i, x in enumerate(sentence.split())}
        self.word2tok = pd.DataFrame.from_dict({'word': word2tok_dict.keys() , 'tokens': word2tok_dict.values()})
        self.get_vocab(sentence)
        
    def get_desired_output(self, sentence):
        idx = 1
        enc =[self.tokenizer.encode(x, add_special_tokens=False) for x in sentence.split()]

        desired_output = []

        for token in enc:
            tokenoutput = []
            for ids in token:
                tokenoutput.append(idx)
                idx +=1
            desired_output.append(tokenoutput)
        
        self.get_word_to_tok(desired_output, sentence)
            
        return desired_output
    
    def get_words_attention(self, attention):
        N = self.word2tok.shape[0]
        d = np.array([[0.] * len(self.tokens) for _ in range(N)])
        p = np.array([[0.] * N for _ in range(N)])
        for idx, token in enumerate(self.tokens):
            for i, word_tok in enumerate(self.word2tok.tokens.values):
                if idx in word_tok and d[i].sum() == 0:
                    d[i] = attention[self.word2tok.loc[i,'tokens']].detach().numpy().sum(axis=0)

        for idx, token in enumerate(self.tokens):
            for i, word_tok in enumerate(self.word2tok.tokens.values):
                if idx in word_tok and p[:,i].sum() == 0:
                    p[:,i] = d[:,self.word2tok.loc[i,'tokens']].sum(axis=1)
        return p

    def get_mapping_dict(self, p):
        N = self.word2tok.shape[0]
        mapping_dict = {}
        for word_idx in range(N):
            if self.word2tok.loc[word_idx, 'word'] == "</s>":
                break
            mask = (p[word_idx] >= self.att_threshold)
            attention_word = p[word_idx][mask]
            map_word = list(self.word2tok.word.values[mask])
            # Проверяем наличие исходного слова, если есть удаляем
            clean_map_word = []
            clean_attention_word = []
            for w in map_word:
                i = map_word.index(w)
                if w not in [self.word2tok.loc[word_idx, 'word'], "</s>"]:
                    clean_map_word.append(w)
            clean_map_word = [self.drop_punc(w) for w in clean_map_word]
            if len(clean_map_word) != 0:
                mapping_dict[self.word2tok.loc[word_idx, 'word']] = clean_map_word

        return mapping_dict
                
    
    def drop_punc(self, word):
        punc = '''!()-[]{};:"\,<>./?@#$%^&*_~'''
        for p in punc:
            word = word.replace(p,'')
        return word

    def make_dict(self, description):
        res = {}
        for i,char in enumerate(description):
            if char == '[':
                begin = i + 1
            if char == ']':
                finish = i
                role = description[begin:finish].split(': ')
                res[role[0]] = role[1]
        return res

    def mapping(self, mapping_dict, eng_words):
        res = ''
        eng_words = eng_words.split()
        for  i, word in enumerate(eng_words):
            try:
                for j, w in enumerate(mapping_dict[word]):
                    if w not in res:
                        res += ' '
                        res += w
            except:
                continue
        return res[1:]

    def result(self, mapping_dict, roles, tat_sentence):
        new_roles = []
        for verb in roles['verbs']:
            result = {}
            srl_verb = {}  

            # Делаем маппинг ролей в виде словаря
            description = verb['description']
            description = self.make_dict(description)
            for srl in description:
                mapped_word = self.mapping(mapping_dict, description[srl])
                srl_verb[srl] = mapped_word
            keys = srl_verb.keys()
            
            # Если нет разметки после мапинга, то пропускаем этот глагол
            if len(keys) == 1 and 'V' in keys:
                continue
            
            # Сохраняем глагол предикат
            new_verb = self.mapping(mapping_dict, verb['verb'])
            # Записывваем тэги на основании словаря ролей
            sentence = tat_sentence.split()
            new_tags = []

            for word in sentence:
                for srl in keys:  
                    if self.drop_punc(word) in srl_verb[srl]:
                        role = srl
                        break
                    else:
                        role = 'O'
                new_tags.append(role)

            result['verb'] = new_verb
            result['description'] = srl_verb
            result['tags'] = new_tags

            new_roles.append(result)
        return new_roles
    

    def predict(self, tat_sentence, source_language):
        eng_sentence = translate(tat_sentence, source_language)[0]
        roles = self.srl_model.predict(sentence=eng_sentence)
        
        sentence = eng_sentence + self.sep_token + tat_sentence
        
        desired_output = self.get_desired_output(sentence)
        
        attention = self.model(
            input_ids=self.tokenized.input_ids,
            attention_mask=self.tokenized.attention_mask,
            output_attentions = True
        )['attentions'][self.att_layer][0][self.att_head]
        
        att = self.get_words_attention(attention)
        mapping_dict = self.get_mapping_dict(att)
        new_roles = {}
        new_roles['verbs'] = self.result(mapping_dict, roles, tat_sentence)
        new_roles['words'] = tat_sentence.split()
        return new_roles

model = Model_srl()

def index(request):
    q = request.GET.get('about' , '')
    if q == 'about':
        context ={}
        return render(request, "about.html", context)
        
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        print("Got values")
        t = ""
        for key, value in request.POST.items():
            t = t + key + value + "\n"
            if key == "phrase":
                result  = value
                print(value)
            if key == "language":
                lang  = value
            
        #result_array = result.split("=")
        #sentence_kz = result_array[0]
        #sentence_en = result_array[1]
        
        
        ret_list = model.predict(result, lang)
        ret = f'{ret_list}'
        s = ''
        try:
            tags = ret_list['verbs'][0]['tags']
            words = ret_list['words']
        
            if len(tags) == len(words):
                for i in range(len(tags)):
                    if tags[i] == 'V':
                        s = s + ' <span title="' + tags[i] + '" style="background:#00eaff;text-decoration-line: underline; text-decoration-style: double">' + words[i] + '</span><sup>' + tags[i] + '</sup>'
                    elif tags[i] == 'ARG0':
                        s = s + ' <span title="' + tags[i] + '" style="background:#beb6ae">' + words[i] + '</span><sup>' + tags[i] + '</sup>'
                    elif tags[i] == 'ARG1':
                        s = s + ' <span title="' + tags[i] + '" style="background:#17d685">' + words[i] + '</span><sup>' + tags[i] + '</sup>'
                    elif tags[i] == 'ARG3':
                        s = s + ' <span title="' + tags[i] + '" style="background:#e1e1e1">' + words[i] + '</span><sup>' + tags[i] + '</sup>'
                    elif tags[i] == 'ARG4':
                        s = s + ' <span title="' + tags[i] + '" style="background:#8024c0">' + words[i] + '</span><sup>' + tags[i] + '</sup>'
                    elif tags[i] == 'ARG2':
                        s = s + ' <span title="' + tags[i] + '" style="background:#ff3885">' + words[i] + '</span><sup>' + tags[i] + '</sup>'
                    elif tags[i] == 'O':
                        s = s + " " + words[i]
                    else:
                        s = s + ' <span title="' + tags[i] + '" style="background:#ffff11">' + words[i] + '</span><sup>' + tags[i] + '</sup>'
        except:
            s = ''



        
        
        context ={}
        context['Debug'] = t
        context['FullValue'] = s
        context['Ret'] = ret

        return render(request, "render.html", context)
        #return HttpResponse("Debug info: \n" + t + "\nGot this value:" + ret + "<br><a href = '/kernel/'>Return to the main page</a>")
    context ={}
    context['form']= InputForm()
    return render(request, "home.html", context)
    #return HttpResponse("Hello, world. You're at the polls index.")

def about(request):
    context ={}
    return render(request, "about.html", context)
