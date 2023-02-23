import json
import os
import re
import random
import numpy as np
from pythonds.basic import Stack
from tqdm import tqdm
from time import time

string_list = [
    # (가)-(하)
    '(가)', '(나)', '(다)', '(라)', '(라)', '(마)', '(바)', '(사)', '(아)', '(자)', '(차)', '(카)', '(타)', '(파)', '(하)',
    # 문제에 등장하는 '인물'의 이름 - 3단계
    '남준', '윤기', '호석', '정국', '유나', '은지', '민영', '유정', '태형', '석진', '지민', '지영', '현정', '슬기', '미란', '호철', '영철',\
    '수호', '성주', '선미', '주리', '정우', '명현', '준호', '명우', '정연', '민성', '요한', '한별', '수민', '지수', '석현', '소미', '정욱',\
    '수진', '슬기', '현정', '미란', '호현', '재영', '지희', '소라', '민수', '정아', '지연', '희수', '미나', '석형', '송화', '진호', '영수',\
    '진수', '수아', '주환', '선호', '제니', '지수', '로희', '지우', '태희', '지훈', '주만', '지용', '나현', '하연', '민지', '주와', '선주',\
    '지혜', '지석', '지선', '나연', '원찬', '정호', '민기', '수현', '수지', '민정', '진영', '철중', '준선', '민호', '명석', '태연', '선우',\
    '다정', '연희',
    # 문제에 등장하는 '인물'의 이름 - 2단계
    '경수', '미라', '민주', '현지', '상민', '윤정', '예원', '영표', '재선', '승연', '승기', '혜수', '가은', '미애', '효리', '준수', '예림',\
    '찬우', '민현', '송희', '은주', '현우', '진희', '건욱', '영인', '혁찬', '연진', '호동', '유승', '신영', '석기', '웅이', '민우', '효진',\
    '은정', '미금', '영미', '수애', '승호', '덕출', '종국', '진석', '미호', '지현', '예슬', '영준', '동국', '희진', '용운', '동균', '현수',\
    '예준', '경환', '빈우', '지완', '정수', '현주', '윤아', '미영', '진아', '정열', '은규', '승찬', '영희', '정희', '철수', '은수', '민준',\
    '동주', '성재', '우준', '종인', '철민', '성민', '민구', '주희', '영훈', '도훈', '기훈', '대현', '승훈', '기윤', '찬휘', '동우', '민엽',\
    '성정', '진형', '진국', '재웅', '은성', '원희', '동일', '진우', '효민', '진섭', '동준', '초원', '영주', '시헌', '영우', '준우', '하은',\
    '재혁', '윤하', '민주', '진주', '찬호', '희철', '수연', '동민', '용희', '미혜', '가영', '소현', '소민', '철이', '현도', '제윤', '승아',\
    '서연', '태훈', '민찬', '승민', '혜인', '낙원', '미연', '영탁', '승환', '나린', '완수', '유경', '민진', '민서', '은설', '수정', '현서',
    # 가족 관계를 나타내는 단어
    '손자', '손녀', '조카', '이모', '삼촌', '동생', '누나', '오빠', '아버지', '어머니', '할머니', '할아버지', '엄마', '아빠', '나', '저', '형', '언니', '고모', 
    # 신체를 나타내는 단어
    '손가락', '발가락', '팔', '다리',
    # 성별을 구분하는 단어
    '암컷', '수컷', '암탉', '수탉', '여학생', '남학생', '여자', '남자',
    # 색을 나타내는 단어
    '흰색', '검은색', '파란색', '노란색', '초록색', '보라색', '노란색', '빨간색', '주황색', '남색', '검정색', '회색', '분홍색', '갈색', '하늘색', '하얀색', '파랑', '하양', '빨강', '노랑', '초록', '보라', '검정', '분홍',
    # 과목을 나타내는 단어
    '영어', '수학', '국어', '사회', '과학', '음악', '미술', '체육',
    # 동물을 나타내는 단어
    '오리', '닭', '토끼', '물고기', '고래', '거위', '달팽이', '개구리', '강아지', '고양이', '비둘기', '병아리', '강아지', '달팽이', '염소', '홍학', '두루미', '꿩', '돼지', '호랑이', '다람쥐', '거북이', '하이에나', '치타',
    # 꽃을 나타내는 단어
    '장미', '백합', '튤립', '카네이션', '국화', '화분', '화단', '꽃병', '채송화', '봉선화', 
    # 운동 관련 단어
    '배구공', '농구공', '축구공', '탁구공', '야구공', '줄넘기', '달리기', '수영', '시합',
    # 음식 관련 단어
    '사과', '배', '감', '귤', '포도', '수박', '참외', '딸기', '복숭아', '바나나', '오렌지', '자두',
    '토마토', '무', '당근', '오이', '배추', '상추', '양상추', '감자', '양파',
    '사탕', '김밥', '빵', '과자', '음료수', '주스', '우유', '달걀', '계란', '단팥빵', '소라빵', '메론', 
    '호박', '가지', '아이스크림', 
    # 학습에 필요한 물품을 나타내는 단어
    '연필', '색연필', '지우개', '공책', '도화지', '색종이', '풀', '테이프', '바둑돌', '구슬', '상자', '나무토막', '장난감', '책장', '책꽂이', '연습장', 
    # 일반적인 장소를 나타내는 단어
    '서점', '마트', '문구점', '집', '학교', '수영장', '교실', '도서관', '박물관', '운동장', '주차장', '정류장', '아파트', '농장', '강당', '경찰서', '소방서', '병원', '약국', '공원', '과수원', '편의점', '수족관', '은행', 
    # 이동수단을 나타내는 단어
    '비행기', '자동차', '트럭', '자전거', '오토바이', '기차', '버스', '엘리베이터', '지하철', '택시', '승용차', '주전자', 
    # 건물 관련 용어
    '페인트', '벽', '천장', '문', '울타리',
    # 그 외 trainset에서 추가
    '초코우유', '딸기우유', '바나나우유', '커피우유', '흰우유', '우산', '지팡이', '수조', '양동이', '접시', '사과파이',
    # 숫자가 들어간 
    '100원', '50원', '10원', '500원', '1000원', '5000원', '10000원', '50000원'
]

josa_list = ['을', '를', '이', '가', '은', '는', '의', '에', '로', '라면', '으로', '와', '과', '에서',
             '까지', '부터', '에게', '보다', '께', '처럼', '이라도', '라도', '으로서', '로서', '한테', 
             '께', '께서', '한테서', '이랑', '랑', '마다', '만큼', '마저', '이나마', '나마']

class Dataset:
    def __init__(self, model_name, data_dir, dataset, add_kor_number, add_figure_number, testsets, use_ixc, use_iec, use_isc):
        self.model_name = model_name
        self.data_dir = data_dir
        self.data_name = dataset
        self.testsets = testsets
        self.add_kor_number = add_kor_number
        self.add_figure_number = add_figure_number
        self.use_ixc = use_ixc
        self.use_iec = use_iec
        self.use_isc = use_isc
        self.mode = 'chall' if 'chall' in self.data_name else 'submit'

        if 'chall' in self.data_name:
            self.load_data_chall(model_name, data_dir)

        # For final submission (dataset/problemsheet.json)
        elif 'dataset' in self.data_name:
            self.load_data_submit(model_name, data_dir)

    def load_data_chall(self, model_name, data_dir):
        # read_json
        train_path = os.path.join(data_dir, self.data_name, 'train.json')
        valid_path = os.path.join(data_dir, self.data_name, 'valid.json')
        with open(train_path, 'r', encoding='utf-8-sig') as f:
            train_json = json.load(f)
        with open(valid_path, 'r', encoding='utf-8-sig') as f:
            valid_json = json.load(f)
        test_paths = [os.path.join(data_dir, self.data_name, f'{test_name}.json') for test_name in self.testsets]
        test_jsons = []
        for test_path in test_paths:
            with open(test_path, 'r', encoding='utf-8-sig') as f:
                test_jsons.append(json.load(f))

        # initializing
        self.idx2question = dict()
        self.idx2solution = dict()
        self.idx2qtype = dict()
        self.idx2level = dict()
        self.idx2category_big = dict()
        self.idx2category_small = dict()
        self.idx2isstring = dict()
        self.idx2INC = dict()
        self.idx2IXC = dict()
        self.idx2IEC = dict()
        self.idx2ISC = dict()
        self.idx2IMQ = dict()
        self.idx2NET = dict()
        self.idx2postfix = dict()
        self.idx2template = dict()

        # TODO: 사람이름, 가나다라,
        self.netvocab2netidx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[OP]': 3}
        self.netidx2netvocab = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]', 3: '[OP]'}
        self.operator2idx = {'[PAD]': 0}
        self.idx2operator = {0: '[PAD]'}
        self.templatetoken2idx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2}
        self.idx2templatetoken = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]'}
        self.kornum2num = {'하나': 1, '둘': 2, '셋': 3, '넷': 4, '다섯': 5, '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9, '열': 10,
                           '한': 1, '두': 2, '세': 3, '네': 4} # 서수
        self.kornum2num4figure = {'일':1, '이':2, '삼': 3, '사': 4, '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9, '십': 10} # 기수
        self.string1_list = [s for s in string_list if len(s) == 1]
        self.string2_list = [s for s in string_list if len(s) == 2]
        self.string3_list = [s for s in string_list if len(s) == 3]
        self.string4_list = [s for s in string_list if len(s) == 4]
        self.string5_list = [s for s in string_list if len(s) == 5]

        # Set train/valid/test ids

        # When using double mask, has to use below 3lines
        # train_datas = len(train_json)
        # for i in range(train_datas, train_datas *2 + 1):
        #     train_json[str(i)] = train_json[str(i-train_datas +1)].copy()
        self.train_ids = self.set_values(train_json, start_idx=0 , use_mask=1)
        self.valid_ids = self.set_values(valid_json, start_idx=1000000)
        self.test_ids = []
        for i, test_json in enumerate(test_jsons):
            test_ids = self.set_values(test_json, start_idx=10000000*(i+1))
            self.test_ids.append(test_ids)

        # Set question type ids
        self.idx2qtype_id = dict()
        map_qtype_id = dict()
        for idx, qtype in self.idx2qtype.items():
            if map_qtype_id.get(qtype) is None:
                map_qtype_id[qtype] = len(map_qtype_id)
            self.idx2qtype_id[idx] = map_qtype_id[qtype]

        # save file for debugging
        self.save_dataloader_to_file(train_json, train_path, start_idx=0)
        self.save_dataloader_to_file(valid_json, valid_path, start_idx=1000000)
        for i, (test_json, test_path) in enumerate(zip(test_jsons, test_paths)):
            self.save_dataloader_to_file(test_json, test_path, start_idx=10000000*(i+1))

    def set_values(self, json, start_idx, use_mask=0):       
        idxes = []
        for json_idx in json.keys():
            idx = int(json_idx) + start_idx
            if self.mode == 'chall':
                equation_op = json[json_idx]['equation_op']
                if 'ans' in equation_op:
                    continue
            idxes.append(idx)

            # question 
            question = json[json_idx]['question']
            
            # u+200b 제거
            question = question.replace(u'\u200b', '')
            # \xa0 제거
            question = question.replace(u'\xa0', ' ')
            
            # 천 단위 표시 콤마 제거
            question = re.sub(r'(\d+),(\d+)', r'\1\2', question)
            
            # [, ], ~ 앞 뒤 공백 삽입
            question = question.replace('~', ' ~ ')
            question = question.replace('[', ' [ ')
            question = question.replace(']', ' ] ')

            # 등식이 아니고 계산 기호가 있는 경우 앞 뒤 공백 삽입
            if len(set(question).intersection('=><≥≤')) == 0 and len(set(question).intersection('+-×/÷')) != 0:
                question = question.replace('+', ' + ')
                question = question.replace('×', ' × ')
                question = question.replace('÷', ' ÷ ')
                
                new_question = []
                for w in question.split():
                    new_question.append(w[0]+w[1:].replace('-', ' - '))
                question = ' '.join(new_question)
                
            self.idx2question[idx] = question
            
            # train, valid, test mode (not submit)
            if self.mode == 'chall':
                # if testset
                if start_idx >= 10000000: 
                    postfix = json[json_idx]['equation_op']
                    # test set에 level, category_big, category_small 추가
                    self.idx2level[idx] = json[json_idx]['level']
                    self.idx2category_big[idx] = json[json_idx]['category_big']
                    self.idx2category_small[idx] = json[json_idx]['category_small']
                else:
                    postfix = json[json_idx]['equation_op']

                self.idx2postfix[idx] = postfix
                try:
                    self.idx2isstring[idx] = (len(re.sub(r'[0-9\[\]A-Za-z_ ]', '', postfix)) > 0)
                except:
                    print("self.idx2isstring[idx] = (len(re.sub(r'[0-9\[\]A-Za-z_ ]', '', postfix)) > 0)")
                    from IPython import embed; embed()
                try:
                    qtype = json[json_idx]['qtype']
                except:
                    qtype = '타입미지정'
                self.idx2qtype[idx] = qtype
                try:
                    solution = json[json_idx]['answer'][0]
                    self.idx2solution[idx] = solution
                except:
                    pass

                # Check if value already exists
                if json[json_idx].get('checked') is True:
                    INC = json[json_idx]['INC']
                    IXC = json[json_idx]['IXC']
                    IEC = json[json_idx]['IEC']
                    ISC = json[json_idx]['ISC']
                    IMQ = json[json_idx]['IMQ']
                    NET = json[json_idx]['NET']
                    template = json[json_idx]['template']
                    self.idx2INC[idx] = INC
                    self.idx2IXC[idx] = IXC
                    self.idx2IEC[idx] = IEC
                    self.idx2ISC[idx] = ISC
                    self.idx2IMQ[idx] = IMQ.strip()
                    self.idx2NET[idx] = NET
                    self.idx2template[idx] = template
                    continue
                else:
                    json[json_idx]['checked'] = False

            
            # 문장 전처리
            new_question = []
            for word in question.strip().split():
                # 직*면체, 정*면체, 정*각형, 직*각형, *면체, *각형
                if self.add_figure_number:
                    try:
                        if '면체' in word or '각형' in word:
                            skip_words = ['직각', '다각형', '다면체', '이등변']
                            flag = False
                            for w in skip_words:
                                if w in word:
                                    flag = True
                            if flag: continue
                            
                            # strip words
                            num_idx = word.find('면체')
                            tmp_num = word if num_idx == -1 else word[:num_idx]
                            num_idx = tmp_num.find('각형')
                            tmp_num = tmp_num if num_idx == -1 else tmp_num[:num_idx]                            
                            num_idx = tmp_num.find('직')
                            tmp_num = tmp_num if num_idx == -1 else tmp_num[num_idx+1:]
                            num_idx = tmp_num.find('정')
                            tmp_num = tmp_num if num_idx == -1 else tmp_num[num_idx+1:]
                            num_idx = tmp_num.find('평행')
                            tmp_num = tmp_num if num_idx == -1 else tmp_num[num_idx+2:]
                            
                            if tmp_num[0] in [str(s) for s in self.kornum2num4figure.values()]:
                                continue
                            
                            # convert to number
                            if len(tmp_num) == 1:
                                new_question.append(str(self.kornum2num4figure[tmp_num]))
                            else:
                                tmp_num = list(tmp_num)
                                whole_num, digit = 0, 1
                                for num in tmp_num[::-1]:
                                    if num in ['십', '백', '천', '만']:
                                        if num == '십':
                                            digit = 10
                                        elif num == '백':
                                            digit = 100
                                        elif num == '천':
                                            digit = 1000
                                        else:
                                            digit = 10000
                                        if num == tmp_num[0]:
                                            whole_num += digit
                                        continue
                                    whole_num += digit * self.kornum2num4figure[num]
                                new_question.append(str(whole_num))
                    except:
                        pass

                # 수사가 등장시 숫자 추가
                if self.add_kor_number and (word in self.kornum2num.keys()):
                    new_question.append(str(self.kornum2num[word]))
                new_question.append(word)

            question = ' '.join(new_question)

            # INC, IMQ, IXC, IEC, ISC
            IMQ = ''
            self.idx2INC[idx] = dict()
            num2INC = dict()
            self.idx2IXC[idx] = dict()
            alpha2IXC = dict()
            self.idx2IEC[idx] = dict()
            eq2IEC = dict()
            self.idx2ISC[idx] = dict()
            str2ISC = dict()

            for word in question.split():
                # 등식이 등장 시 IEC 부여
                if ('=' in word or '<' in word or '>' in word or '≥' in word or '≤' in word) and self.use_iec:
                    eq = ''
                    for c in word:
                        if c in '1234567890+-×/÷*%.=><≥≤ABCDEFGHIJKLMNOPQRSTUVWXYZ()':
                            eq += c
                        else:  # 88점입니다.
                            break
                    IEC = '[E' + str(len(self.idx2IEC[idx])) + ']'
                    self.idx2IEC[idx][IEC] = eq
                    eq2IEC[eq] = IEC
                    IMQ += IEC + ' '

                # 영어 대문자가 등장시 IXC 부여   
                elif self.use_ixc and (len(set(word).intersection('ABCDEFGHIJKLMNOPQRSTUVWXYZ')) > 0):
                    unk = re.sub(r'[^0-9A-Z]', '', word)
                    if alpha2IXC.get(unk) is not None:
                        IXC = alpha2IXC[unk]
                    else:
                        IXC = '[X' + str(len(self.idx2IXC[idx])) + ']'
                        self.idx2IXC[idx][IXC] = unk
                        alpha2IXC[unk] = IXC
                    IMQ += IXC + ' '

                # 숫자가 등장시 INC 부여
                elif word[0].isdigit() or ((word[0] == '-' and len(word) != 1) and word[1].isdigit()):
                    num = ''
                    # 1,000원 -> N: 1000
                    for c in re.sub('[,]', '', word):
                        if c in '1234567890./-':  # 소수, 분수, 음수 고려
                            num += c
                        else:  # 88점입니다.
                            break
                    INC = '[N' + str(len(self.idx2INC[idx])) + ']'
                    self.idx2INC[idx][INC] = num
                    num2INC[num] = INC
                    IMQ += INC + ' '

                # 정답식과 문제에 특정 문자열이 등장시 ISC 부여
                # 특정 문자열이 등장시 ISC 부여
                josa_string = ''.join(josa_list)
                if self.use_isc and ((re.sub(f'[,{josa_string}]', '', word) in self.string1_list) or (word[:2] in self.string2_list) or (word[:3] in self.string3_list) or (word[:4] in self.string4_list) or (word[:5] in self.string5_list)):
                    if word[:5] in self.string5_list:
                        tmp_str = word[:5]
                    elif word[:4] in self.string4_list:
                        tmp_str = word[:4]
                    elif word[:3] in self.string3_list:
                        tmp_str = word[:3]
                    elif word[:2] in self.string2_list:
                        tmp_str = word[:2]
                    elif re.sub(f'[,{josa_string}]', '', word) in self.string1_list:
                        tmp_str = re.sub(f'[,{josa_string}]', '', word)

                    if str2ISC.get(tmp_str) is not None:
                        ISC = str2ISC[tmp_str]
                    else:
                        ISC = '[S' + str(len(self.idx2ISC[idx])) + ']'
                        self.idx2ISC[idx][ISC] = tmp_str
                        str2ISC[tmp_str] = ISC
                    IMQ += ISC + ' '

                IMQ += word + ' '
            
            
            # # Implement mask to input question, if use_mask==1
            # if use_mask == 1:
            #     IMQ = IMQ.strip()
            #     IMQ = IMQ.split(' ')
            #     maskwordsidx = random.sample(range(len(IMQ)), (len(IMQ)//15 + 1))

            #     for i in maskwordsidx:
            #         while(True):
            #             if '[' not in IMQ[i] :
            #                 IMQ[i] = '[MASK]'
            #                 break
            #             else:
            #                 i = i+1
            #                 if i == len(IMQ):
            #                     i=0
            #         self.idx2IMQ[idx] = (' ').join(IMQ)
            # else:
            self.idx2IMQ[idx] = IMQ.strip()

            if self.mode == 'chall':
                # postfix -> NET (For TM-generation)
                NET = postfix.split()
                                
                # Number -> INC
                for k, v in self.idx2INC[idx].items():
                    for NET_idx, token in enumerate(NET):
                        if v == token:
                            NET[NET_idx] = k
                # 미지수 -> IXC
                for k, v in self.idx2IXC[idx].items():
                    for NET_idx, token in enumerate(NET):
                        if v == token:
                            NET[NET_idx] = k
                # 등식 -> IEC
                for k, v in self.idx2IEC[idx].items():
                    for NET_idx, token in enumerate(NET):
                        if v == token:
                            NET[NET_idx] = k
                # 문자열 -> ISC
                for k, v in self.idx2ISC[idx].items():
                    for NET_idx, token in enumerate(NET):
                        if v == token:
                            NET[NET_idx] = k
                # Constant -> C
                for NET_idx, token in enumerate(NET):
                    try:
                        if token in '+-*/><=' or token[0].isdigit() or (token[0] == '-' and token[1].isdigit()):
                            NET[NET_idx] = '[C' + token + ']'
                    except:
                        print("if token[0].isdigit() or (token[0] == '-' and token[1].isdigit()) or token in '><':                          NET[NET_idx] = '[C' + token + ']'")
                        from IPython import embed; embed()
                        exit()
                # Operation -> OP & Constant 처리
                for NET_idx, token in enumerate(NET):
                    if token.startswith('[OP'):
                        if self.operator2idx.get(token) is None:
                            self.operator2idx[token] = len(self.operator2idx)
                            self.idx2operator[self.operator2idx[token]] = token
                        NET[NET_idx] = '[OP]'
                    else:
                        if self.netvocab2netidx.get(token) is None:
                            self.netvocab2netidx[token] = len(self.netvocab2netidx)
                            self.netidx2netvocab[self.netvocab2netidx[token]] = token
                # for NET_idx, token in enumerate(NET):
                #     if self.netvocab2netidx.get(token) is None:
                #         self.netvocab2netidx[token] = len(self.netvocab2netidx)
                #         self.netidx2netvocab[self.netvocab2netidx[token]] = token
                #     if token.startswith('[OP'):
                #         if self.operator2idx.get(token) is None:
                #             self.operator2idx[token] = len(self.operator2idx)
                #             self.idx2operator[self.operator2idx[token]] = token
                #         NET[NET_idx] = token
                self.idx2NET[idx] = ' '.join(NET)

                # postfix -> template (For GEO)
                template = postfix.split()
                for k, v in self.idx2INC[idx].items():
                    for template_idx, token in enumerate(template):
                        if v == token:
                            template[template_idx] = k
                # 미지수 -> IXC
                for k, v in self.idx2IXC[idx].items():
                    for template_idx, token in enumerate(template):
                        if v == token:
                            template[template_idx] = k
                # 등식 -> IEC
                for k, v in self.idx2IEC[idx].items():
                    for template_idx, token in enumerate(template):
                        if v == token:
                            template[template_idx] = k
                # 문자열 -> ISC
                for k, v in self.idx2ISC[idx].items():
                    for template_idx, token in enumerate(template):
                        if v == token:
                            template[template_idx] = k
                # Constant -> C
                for template_idx, token in enumerate(template):
                    if token in '+-*/><=' or token[0].isdigit() or (token[0] == '-' and token[1].isdigit()):
                        template[template_idx] = '[C' + token + ']'
                # templatetoken dict에 추가
                for template_idx, token in enumerate(template):
                    if self.templatetoken2idx.get(token) is None:
                        self.templatetoken2idx[token] = len(self.templatetoken2idx)
                        self.idx2templatetoken[self.templatetoken2idx[token]] = token
                self.idx2template[idx] = ' '.join(template)
                
        return np.array(idxes)


    def load_data_submit(self, model_name, data_dir):
        test_path = os.path.join(self.data_name, 'problem_sheet.json')
        with open(test_path, 'r', encoding='utf-8-sig') as f:
            test_json = json.load(f)
            
        test_json = test_json['problem_sheet'] 
        
        # initializing
        self.idx2question = dict()
        self.idx2INC = dict()
        self.idx2IXC = dict()
        self.idx2IEC = dict()
        self.idx2ISC = dict()
        self.idx2IMQ = dict()

        # TODO: 사람이름, 가나다라
        self.netvocab2netidx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[OP]': 3}
        self.netidx2netvocab = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]', 3: '[OP]'}
        self.operator2idx = {'[PAD]': 0}
        self.idx2operator = {0: '[PAD]'}
        self.templatetoken2idx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2}
        self.idx2templatetoken = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]'}
        self.kornum2num = {'하나': 1, '둘': 2, '셋': 3, '넷': 4, '다섯': 5, '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9, '열': 10,
                           '한': 1, '두': 2, '세': 3, '네': 4} # 서수
        self.kornum2num4figure = {'일':1, '이':2, '삼': 3, '사': 4, '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9, '십': 10} # 기수
        
        self.string1_list = [s for s in string_list if len(s) == 1]
        self.string2_list = [s for s in string_list if len(s) == 2]
        self.string3_list = [s for s in string_list if len(s) == 3]
        self.string4_list = [s for s in string_list if len(s) == 4]
        self.string5_list = [s for s in string_list if len(s) == 5]

        # Set train/valid/test ids
        self.test_ids = self.set_values(test_json, start_idx=0)
        
        # Save preprocessed files
        self.save_dataloader_to_file(test_json, os.path.join('problem_sheet_preprocessed.json'), start_idx=0)


    def load_data_CC(self, model_name, data_dir, dataset):

        # read_json
        data_path = os.path.join(data_dir, self.data_name, 'questions.json')
        with open(data_path, 'r') as f:
            all_json = json.load(f)

        # Set train/valid/test ids
        all_ids = np.arange(len(all_json))
        np.random.shuffle(all_ids)
        self.train_ids = all_ids[:int(0.7 * len(all_ids))]
        self.valid_ids = all_ids[int(0.7 * len(all_ids)): int(0.8 * len(all_ids))]
        self.test_ids = all_ids[int(0.8 * len(all_ids)):]

        # initializing
        self.idx2question = dict()
        self.idx2alignment = dict()
        self.idx2solution = dict()
        self.idx2equation = dict()
        self.idx2INC = dict()
        self.idx2IMQ = dict()
        self.idx2NET = dict()
        self.idx2postfix = dict()

        # TODO: Constant 고려 필요 (예정)
        self.netvocab2netidx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, 'OP': 3}
        self.netidx2netvocab = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]', 3: 'OP'}
        self.operator2idx = {'[PAD]': 0}
        self.idx2operator = {0: '[PAD]'}

        # Set Values using json
        for i in range(len(all_json)):
            idx = all_json[i]['iIndex']

            # question, alignment, solution, equation
            question = all_json[i]['sQuestion']
            alignments = all_json[i]['lAlignments']
            solution = all_json[i]['lSolutions'][0]
            equation = all_json[i]['lEquations'][0]
            self.idx2question[idx] = question
            self.idx2alignment[idx] = alignments
            self.idx2solution[idx] = solution
            self.idx2equation[idx] = equation

            # INC, IMQ
            self.idx2INC[idx] = dict()
            IMQ = ''
            num2INC = dict()
            for word in question.split():
                # 숫자가 등장시 INC 부여
                re_word = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', word)
                if re_word.isdigit():
                    INC = 'N' + str(len(self.idx2INC[idx]))
                    self.idx2INC[idx][INC] = re_word
                    num2INC[re_word] = INC
                    IMQ += INC + ' '
                IMQ += word + ' '
            self.idx2IMQ[idx] = IMQ.strip()

            # infix -> postfix
            postfix = self.infixToPostfix(self.make_space_eq(equation[2:]))
            self.idx2postfix[idx] = postfix

            # postfix -> NET
            NET = postfix.split()
            for k, v in self.idx2INC[idx].items():
                for NET_idx, token in enumerate(NET):
                    if v+'.0' == token:
                        NET[NET_idx] = k
                        break
            for NET_idx, token in enumerate(NET):
                if token in '+-*/':
                    if self.operator2idx.get(token) is None:
                        self.operator2idx[token] = len(self.operator2idx)
                        self.idx2operator[self.operator2idx[token]] = token
                    NET[NET_idx] = 'OP'
                else:
                    if self.netvocab2netidx.get(token) is None:
                        self.netvocab2netidx[token] = len(self.netvocab2netidx)
                        self.netidx2netvocab[self.netvocab2netidx[token]] = token
            self.idx2NET[idx] = ' '.join(NET)

    def infixToPostfix(self, infixexpr):
        prec = {}
        prec["*"] = 3
        prec["/"] = 3
        prec["+"] = 2
        prec["-"] = 2
        prec["("] = 1
        opStack = Stack()
        postfixList = []
        tokenList = infixexpr.split()

        for token in tokenList:
            if re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', token).isdigit():
                postfixList.append(token)
            elif token == '(':
                opStack.push(token)
            elif token == ')':
                topToken = opStack.pop()
                while topToken != '(':
                    postfixList.append(topToken)
                    topToken = opStack.pop()
            else:
                while (not opStack.isEmpty()) and (prec[opStack.peek()] >= prec[token]):
                    postfixList.append(opStack.pop())
                opStack.push(token)

        while not opStack.isEmpty():
            postfixList.append(opStack.pop())
        return " ".join(postfixList)

    def make_space_eq(self, infix):
        new_infix = ''

        for c in infix:
            if c in '+-*/()=%':
                new_infix += ' ' + c + ' '
            else:
                new_infix += c
        return new_infix

    def __str__(self):
        ret_str = '\n'
        ret_str += 'Dataset: %s\n' % self.data_name
        # ret_str += '# of docs_data: %d\n' % len(self.docs_data)
        # ret_str += '# of rels_data: %d(%d+%d)\n' % (self.num_rels_train + self.num_rels_test, self.num_rels_train, self.num_rels_test)
        return ret_str

    def save_dataloader_to_file(self, orig_json, data_path, start_idx=0):
        for json_idx in orig_json.keys():
            idx = int(json_idx)+start_idx
            orig_json[json_idx]['INC'] = self.idx2INC.get(idx)
            orig_json[json_idx]['IXC'] = self.idx2IXC.get(idx)
            orig_json[json_idx]['IEC'] = self.idx2IEC.get(idx)
            orig_json[json_idx]['ISC'] = self.idx2ISC.get(idx)
            orig_json[json_idx]['IMQ'] = self.idx2IMQ.get(idx)
            if self.mode == 'chall':
                orig_json[json_idx]['NET'] = self.idx2NET.get(idx)
                orig_json[json_idx]['template'] = self.idx2template.get(idx)
