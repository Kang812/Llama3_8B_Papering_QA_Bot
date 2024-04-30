import pandas as pd
import json
from tqdm import tqdm

def data_processing(dataframe_path, save_path):
    json_data = []
    data = pd.read_csv(dataframe_path)
    
    for _, row in tqdm(data.iterrows()):
        question_1 = row['질문_1']
        question_2 = row['질문_2']

        for q in [question_1, question_2]:
            for a in range(1, 6):
                answer = row[f'답변_{a}']
                json_data.append({
                    "question": q,
                    "answer" : answer
                })
        
    json_string = json.dumps(json_data, ensure_ascii=False, indent=4)

    with open(save_path, "w", encoding='utf-8') as f:
        f.write(json_string)

if __name__ == '__main__':
    dataframe_path = "/workspace/papering_qa/data/train.csv"
    save_path = "/workspace/papering_qa/data/train.json"
    data_processing(dataframe_path, save_path)