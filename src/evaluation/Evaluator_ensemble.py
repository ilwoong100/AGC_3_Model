from email import header
import os
import re
import json
import numpy as np

from collections import OrderedDict, Counter
from pythonds.basic import Stack
from sklearn.metrics import accuracy_score
from IPython import embed
from evaluation.PostfixConverter import PostfixConverter

class Evaluator_ensemble:
    def __init__(self, early_stop_measure='acc_equation'):
        self.early_stop_measure = early_stop_measure
        self.pf_converter = PostfixConverter()

    def evaluate(self, model, dataset, mode='valid', ensemble_type='max_voting'):
        if mode == 'valid':
            eval_id = dataset.valid_ids
        elif 'test' in mode:
            test_num = dataset.testsets.index(mode)
            eval_id = dataset.test_ids[test_num]
        elif mode in ['submit', 'debug']:
            eval_id = dataset.test_ids

        # Get score
        if mode in ['submit', 'debug']:
            self.get_score_submission(model, dataset, mode, eval_id)
        else:
            score = self.get_score(model, dataset, mode, eval_id, ensemble_type)
            return score

    # ------------------ SUBMISSION ------------------ #
    def get_score_submission(self, model, dataset, mode, eval_id, ensemble_type='max_voting'):
        from urllib import request
        
        # list 형태로 여러 모델을 넣어줄 경우, 앙상블 방식으로 평가
        if isinstance(model, list):
            test_num = f'{ensemble_type}_{len(model)}models'
            # max voting 방식
            if ensemble_type == 'max_voting':
                final_pred_eq = []
                pred_eq_list = []
                for m in model:
                    ans, eq, loss = m.predict(mode, self.pf_converter)
                    pred_eq_list.append(eq)
                
                pred_eq_array = np.array(pred_eq_list)
                num_pred_eq = pred_eq_array.shape[1]
                for i in range(num_pred_eq):
                    temp = pred_eq_array[:, i]
                    if len(set(Counter(temp).values())) == 1:
                        # 첫 번째 모델의 결과를 사용 (성능 1위 모델을 맨앞에 두기)
                        final_pred_eq.append(temp[0])
                    else:
                        max_voted_eq = Counter(temp).most_common(1)[0][0]
                        final_pred_eq.append(max_voted_eq)

                eval_equation = final_pred_eq
                eval_loss = loss
                eval_answer = ans

            # 필요 시 다른 앙상블 방식 구현
            else:
                eval_answer, eval_equation, eval_loss = model[0].predict(mode, self.pf_converter)
            # 아래 log_dir를 사용하기 위해 첫 번째 모델 saves에 저장
            model = model[0]

        # 모델 하나일 때 평가
        else:
            # get answer and equation
            eval_answer, eval_equation, _ = model.predict(mode, self.pf_converter)

        eval_equation_answer = []
        for idx, eq in enumerate(eval_equation):
            try:
                result, code_string = self.pf_converter.convert(eq, dataset.idx2question[eval_id[idx]])
                eval_equation_answer.append(result)
            except:
                eval_equation_answer.append(6)

        # int, float to .2f
        np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})
        eval_equation_answer = [f"{ans}" for ans in eval_equation_answer]

        
        if mode != 'debug':
            api_url = os.environ['REST_ANSWER_URL'] # REST URL load
        
        # answer template
        json_data = {
            "team_id": "zxcvxd",
            "secret": "7svCQKFCDhjd7CrH",
            "answer_sheet": {}
        }
        
        for i, idx in enumerate(eval_id):
            temp = {
                    "no": f"{idx}", 
                    "answer": f"{eval_equation_answer[i]}"
            }
            json_data['answer_sheet'] = temp
            
            if mode != 'debug':
                # Post to API server
                data = json.dumps(json_data).encode('utf-8')
                req =  request.Request(api_url, data=data)
                # check API server return
                resp = request.urlopen(req)
                
                # check POST result
                resp_json = eval(resp.read().decode('utf-8'))
                print("received message: "+resp_json['msg'])

                if "OK" == resp_json['status']:
                    print("data requests successful!!")
                elif "ERROR" == resp_json['status']:    
                    raise ValueError("Receive ERROR status. Please check your source code.")     
                
        
        print("Inference Done!!")
        
        if mode != 'debug':
            # request end of mission message
            message_structure = {
                "team_id": "zxcvxd",
                "secret": "7svCQKFCDhjd7CrH",
                "end_of_mission": "true"
            }
            # json dump & encode utf-8
            tmp_message = json.dumps(message_structure).encode('utf-8')
            request_message = request.Request(api_url, data=tmp_message) 
            resp = request.urlopen(request_message) # POST

            resp_json = eval(resp.read().decode('utf-8'))
            print("received message: "+resp_json['msg'])

            if "OK" == resp_json['status']:
                print("data requests successful!!")
            elif "ERROR" == resp_json['status']:    
                raise ValueError("Receive ERROR status. Please check your source code.")  
        else:
            answer_dict = {}
            for i, idx in enumerate(eval_id):
                answer_dict[str(idx)] = {}
                answer_dict[str(idx)]['answer'] = eval_equation_answer[i]
            print(answer_dict)
    # ------------------ SUBMISSION ------------------ #

    def get_score(self, model, dataset, mode, eval_id, test_num=None, ensemble_type='max_voting'):
        score = OrderedDict()
        # 실제 정답
        true_answer = []
        for idx in eval_id:
            try:
                true_answer.append(self.pf_converter.convert(dataset.idx2postfix[idx], dataset.idx2question[idx])[0])
            except:
                true_answer.append(6)
                print(f"{mode}", idx, dataset.idx2postfix[idx], "is Errored!!!")
        
        # list 형태로 여러 모델을 넣어줄 경우, 앙상블 방식으로 평가
        if isinstance(model, list):
            test_num = f'{ensemble_type}_{len(model)}models'
            # max voting 방식
            if ensemble_type == 'max_voting':
                final_pred_eq = []
                pred_eq_list = []
                for m in model:
                    ans, eq, loss = m.predict(mode, self.pf_converter)
                    pred_eq_list.append(eq)
                
                pred_eq_array = np.array(pred_eq_list)
                num_pred_eq = pred_eq_array.shape[1]
                for i in range(num_pred_eq):
                    temp = pred_eq_array[:, i]
                    if len(set(Counter(temp).values())) == 1:
                        # 첫 번째 모델의 결과를 사용 (성능 1위 모델을 맨앞에 두기)
                        final_pred_eq.append(temp[0])
                    else:
                        max_voted_eq = Counter(temp).most_common(1)[0][0]
                        final_pred_eq.append(max_voted_eq)

                eval_equation = final_pred_eq
                eval_loss = loss
                eval_answer = ans

            # 필요 시 다른 앙상블 방식 구현
            else:
                eval_answer, eval_equation, eval_loss = model[0].predict(mode, self.pf_converter)
            # 아래 log_dir를 사용하기 위해 첫 번째 모델 saves에 저장
            model = model[0]

        # 모델 하나일 때 평가
        else:
            # get answer and equation
            eval_answer, eval_equation, eval_loss = model.predict(mode, self.pf_converter)

        eval_equation_answer = []
        error_list = []
        for idx, eq in enumerate(eval_equation):
            try: 
                eval_equation_answer.append(self.pf_converter.convert(eq, dataset.idx2question[eval_id[idx]])[0])
                error_list.append(0)
            except:
                eval_equation_answer.append(6)
                error_list.append(1)
        num_error = np.sum(error_list)

        # calculate score
        score[f'{mode}_loss'] = np.mean(eval_loss)
        if eval_answer is not None:
            score['acc_ans'] = accuracy_score(true_answer, eval_answer)
        
        # int, float to .2f
        np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})
        true_answer = [f"{ans}" for ans in true_answer]
        eval_equation_answer = [f"{ans}" for ans in eval_equation_answer]
        score[f'{mode}_accuracy'] = accuracy_score(true_answer, eval_equation_answer)
        score[f'{mode}_num_error'] = num_error
        score[f'{mode}_error_rate'] = num_error / len(eval_equation_answer)

        # Save Predicted
        out_lines = []
        out_lines.append(f"Accuacy: {round(accuracy_score(true_answer, eval_equation_answer)*100, 2)} ({(np.array(true_answer)==np.array(eval_equation_answer)).sum()}/{len(true_answer)}), num_error: {num_error}, error_rate: {round(num_error / len(eval_equation_answer)*100, 2)}")
        out_lines.append(f"\n")

        # calculate score for question type
        type2true = {type:[] for type in dataset.idx2qtype.values()}
        type2pred = {type:[] for type in dataset.idx2qtype.values()}
        for i, idx in enumerate(eval_id):
            type2true[dataset.idx2qtype[idx]].append(true_answer[i])
            type2pred[dataset.idx2qtype[idx]].append(eval_equation_answer[i])

        type_set = ['산술', '수찾기-1', '수찾기-2', '수찾기-3', '순서정하기', '크기비교', '도형', '조합']
        for type in (type_set):
            if len(type2pred[type]) != 0:
                out_lines.append(f"{type}: {round(accuracy_score(type2true[type], type2pred[type])*100, 2)} ({(np.array(type2true[type])==np.array(type2pred[type])).sum()}/{len(type2pred[type])})")

        out_lines.append(f"\n")
        out_lines.append(f"Accuacy: {round(accuracy_score(true_answer, eval_equation_answer)*100, 2)} ({(np.array(true_answer)==np.array(eval_equation_answer)).sum()}/{len(true_answer)})")
        out_lines.append(f"\n")

        if mode == 'valid':
            for i, idx in enumerate(eval_id):
                out_lines.append(f"Index: {idx}")
                out_lines.append(f"Question type: {dataset.idx2qtype[idx]}")
                out_lines.append(f"Question: {dataset.idx2question[idx]}")
                out_lines.append(f"True_postfix: {dataset.idx2postfix[idx]} --> {true_answer[i]}")
                out_lines.append(f"Pred_postfix: {eval_equation[i]} --> {eval_equation_answer[i]}")
                out_lines.append(f"Correct: {true_answer[i] == eval_equation_answer[i]}")
                out_lines.append(f"Error: {True if error_list[i] else False}")
                out_lines.append("\n")
            with open(os.path.join(model.log_dir, f'{mode}_Results_{test_num}.txt'),'w') as f:
                f.write('\n'.join(out_lines))

        if mode.startswith('test'):
            # calculate score for question level
            level2true = {level:[] for level in dataset.idx2level.values()}
            level2pred = {level:[] for level in dataset.idx2level.values()}
            for i, idx in enumerate(eval_id):
                level2true[dataset.idx2level[idx]].append(true_answer[i])
                level2pred[dataset.idx2level[idx]].append(eval_equation_answer[i])

            for level in sorted(list(set(dataset.idx2level.values()))):
                out_lines.append(f"Level {level}: {round(accuracy_score(level2true[level], level2pred[level])*100, 2)} ({(np.array(level2true[level])==np.array(level2pred[level])).sum()}/{len(level2pred[level])})")
            out_lines.append(f"\n")

            # calculate score for category big
            category_big2true = {level:[] for level in dataset.idx2category_big.values()}
            category_big2pred = {level:[] for level in dataset.idx2category_big.values()}
            for i, idx in enumerate(eval_id):
                category_big2true[dataset.idx2category_big[idx]].append(true_answer[i])
                category_big2pred[dataset.idx2category_big[idx]].append(eval_equation_answer[i])

            for level in sorted(list(set(dataset.idx2category_big.values()))):
                out_lines.append(f"Category Big {level}: {round(accuracy_score(category_big2true[level], category_big2pred[level])*100, 2)} ({(np.array(category_big2true[level])==np.array(category_big2pred[level])).sum()}/{len(category_big2pred[level])})")
            out_lines.append(f"\n")

            # calculate score for category small
            category_small2true = {level:[] for level in dataset.idx2category_small.values()}
            category_small2pred = {level:[] for level in dataset.idx2category_small.values()}
            for i, idx in enumerate(eval_id):
                category_small2true[dataset.idx2category_small[idx]].append(true_answer[i])
                category_small2pred[dataset.idx2category_small[idx]].append(eval_equation_answer[i])

            for level in sorted(list(set(dataset.idx2category_small.values()))):
                out_lines.append(f"Category Small {level}: {round(accuracy_score(category_small2true[level], category_small2pred[level])*100, 2)} ({(np.array(category_small2true[level])==np.array(category_small2pred[level])).sum()}/{len(category_small2pred[level])})")
            out_lines.append(f"\n")
        
            for i, idx in enumerate(eval_id):
                out_lines.append(f"Index: {idx}")
                out_lines.append(f"Question type: {dataset.idx2qtype[idx]}")
                out_lines.append(f"Level: {dataset.idx2level[idx]}")
                out_lines.append(f"Category_big: {dataset.idx2category_big[idx]}")
                out_lines.append(f"Category_small: {dataset.idx2category_small[idx]}")
                out_lines.append(f"Question: {dataset.idx2question[idx]}")
                out_lines.append(f"True_postfix: {dataset.idx2postfix[idx]} --> {true_answer[i]}")
                out_lines.append(f"Pred_postfix: {eval_equation[i]} --> {eval_equation_answer[i]}")
                out_lines.append(f"Correct: {true_answer[i] == eval_equation_answer[i]}")
                out_lines.append(f"Error: {True if error_list[i] else False}")
                out_lines.append("\n")
            with open(os.path.join(model.log_dir, f'{mode}_Results_{test_num}.txt'),'w') as f:
                f.write('\n'.join(out_lines))

        return score


def eval_postfix(postfixExpr):
    operandStack = Stack()
    tokenList = postfixExpr.split()

    for token in tokenList:
        if re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', token).isdigit():
            operandStack.push(float(token))
        else:
            if operandStack.size() < 2: return 0
            operand2 = operandStack.pop()
            operand1 = operandStack.pop()
            result = doMath(token,operand1,operand2)
            operandStack.push(result)
    return operandStack.pop()

def doMath(op, op1, op2):
    if op == "*":
        return op1 * op2
    elif op == "/":
        return op1 / op2
    elif op == "+":
        return op1 + op2
    else:
        return op1 - op2