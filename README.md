# DIAL_MWP_model
기학습된 언어모델(Pretrained-Language-Model)을 활용하여 자연어로 이루어진 한글 수학 서술형 문제 (Math Word Problem, MWP)를   푸는 문제입니다.

본 연구는 인공지능산업원천기술개발 사업으로 2022년 1월 1일부터 2022년 12월 31일까지 12개월간 진행된 연구입니다.

## 목표
본 연구에서는 한글로 이루어진 수학 서술형 문제(MWP)를 푸는 심층신경망 모델을 개발합니다.

Rule-based 방식으로 symbol형태로 전환 후 이를 이용하여 본 연구진이 제안한 Operation Equation(풀이과정)을 예측하여 정답을 도출하는 방식을 사용합니다.

## 설치

    pip install -r requirements.txt

## 실행방법

    CUDA_VISIBLE_DEVICES=0 python main.py --dataset "예시 data경로 chall/base_bt" --model_name TM_Generation_1step --seed 0 --gpu 0

## 데이터셋
data폴더에 학습 및 평가 데이터를 넣어주셔야 합니다.

사용한 데이터는 [DIAL_MWP_DATASET](https://github.com/ilwoong100/DIAL_MWP-Dataset)에서 확인할 수 있습니다.
