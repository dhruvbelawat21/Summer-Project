import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import torch
from string import Template
from pathlib import Path
import json
from tqdm.auto import tqdm
import warningsac
warnings.simplefilter("ignore")
import torch


device = "cuda:0" if torch.cuda.is_available() else "cpu"

with open("/content/TeleQnA_training.txt", "r") as f:
    train_data = f.read()
with open("/content/TeleQnA_training.txt", "r") as f:
    test_data = f.read()

train_data=json.loads(train_data)
test_data=json.loads(test_data)



def dict_to_frame(text):
    data = []
    for question_id in list(text.keys()):
        question_details=text[question_id]
        data.append({
           "question_id": question_id,
           "question": question_details["question"],
           "option_1": question_details["option 1"],
           "option_2": question_details["option 2"],
           "option_3": question_details.get("option 3", None),
           "option_4": question_details.get("option 4", None),
           "category": question_details["category"],
         })
    return pd.DataFrame(data)
test_df=dict_to_frame(test_data)
train_df=dict_to_frame(train_data)
test_df['question_id']=test_df['question_id'].str.replace('question ', '').astype(int)
train_df



preamble = \
   'Answer the following question by selecting the most likely answer choice (A, B, C, or D): please generate only answer choice'
template = Template('$preamble\n\n$prompt\n\nA) $a\nB) $b\nC) $c\nD) $d')



def format_input(df, idx):

    prompt = df.loc[idx, 'question']
    a = df.loc[idx, 'option_1']
    b = df.loc[idx, 'option_2']
    c = df.loc[idx, 'option_3']
    d = df.loc[idx, 'option_4']


    input_text = template.substitute(
        preamble=preamble, prompt=prompt, a=a, b=b, c=c, d=d)

    return input_text




print(format_input(test_df, 3))



ans=pd.read_csv('/content/Q_A_ID_training.csv')


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)


def extract_answer(model_output):
    lines = model_output.splitlines()
    for line in lines:
        try:
            if line.startswith("Answer:"):
                return line.split()[1]  # Split by space and return the second word (answer choice)
        except:
            return 'D'  # Answer not found


if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
'''
def get_ans():
    ans_lst=[]
    for i in tqdm(range(test_df.shape[0])):
        inputs = tokenizer(format_input(test_df, i), return_tensors="pt", return_attention_mask=False).to(device)
        outputs = model.generate(**inputs,pad_token_id=tokenizer.pad_token_id, max_length=200)
        text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        ans=extract_answer(text)[0]
        ans_lst.append(ans)
    return ans_lst
'''
def get_ans():
    ans_lst=[]
    for i in tqdm(range(test_df.shape[0])):
        inputs = tokenizer(format_input(test_df, i), return_tensors="pt", return_attention_mask=False).to(device)
        # Increase max_length or better yet, use max_new_tokens to control the generated sequence length
        outputs = model.generate(**inputs,pad_token_id=tokenizer.pad_token_id, max_new_tokens=50)  
        text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        ans=extract_answer(text)[0]
        ans_lst.append(ans)
    return ans_lst
ans_lst=get_ans()


map_ans={'A':1, 'B':2, 'C':3, 'D':4}


ss=pd.read_csv('/content/SampleSubmission.csv')
sub=pd.DataFrame({'Question_ID':test_df['question_id'].values, 'Answer_ID':ans_lst})
sub['Answer_ID']=sub['Answer_ID'].map(map_ans)
sub['Answer_ID']=sub['Answer_ID'].fillna(3).astype(int)
subzindi=pd.concat([sub, ss[~ss['Question_ID'].isin(sub['Question_ID'])]])
subzindi.to_csv('subllm.csv', index=False)







print(ans_lst)
print(len(ans_lst))
print(range(test_df.shape[0]))
