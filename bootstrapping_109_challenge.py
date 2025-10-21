import json
import openai
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_distances

client = openai.OpenAI(api_key="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")

def get_response():
    with open("final_gender_bias_diagnosis_results.json", "r") as one_datafile:
        one_data = json.load(one_datafile)
    data_vector = []
    for entry in one_data:
        response_dict = {}
        response_dict["gender"]= entry["gender"]
        response_dict["response"]= entry["response"]
        data_vector.append(response_dict)
    
    with open("two_final_gender_bias_diagnosis_results.json", "r") as two_datafile:
        two_data = json.load(two_datafile)
    for entry in two_data:
        response_dict = {}
        response_dict["gender"]= entry["gender"]
        response_dict["response"]= entry["response"]
        data_vector.append(response_dict)
    return data_vector 

def get_embeddings(responses, model="text-embedding-3-large"):
    embeddings = []
    for response in responses:
        embedded_resp = client.embeddings.create(input=[response], model=model)
        embeddings.append(embedded_resp.data[0].embedding)
        print("embedded response")
    return embeddings

def embeddings_dist(group_one, group_two):
    return np.mean(cosine_distances(group_one, group_two))

def bootstrap():
    trials = get_response()
    f_group = []
    m_group = []
    for trial in trials:
        if trial["gender"] == "female":
            f_group.append(trial["response"])
        else:
            m_group.append(trial["response"])
    f_embedded = get_embeddings(f_group)
    m_embedded = get_embeddings(m_group)

    control_dist = embeddings_dist(f_embedded, m_embedded)
    all_responses = f_embedded + m_embedded
    cos_distances = []
    for i in range(1, 10001):
        group_one = []
        group_two = []
        for j in range(int(len(all_responses)/2)):
            ind_one = random.randint(0, len(all_responses)-1)
            ind_two = random.randint(0, len(all_responses)-1)
            group_one.append(all_responses[ind_one])
            group_two.append(all_responses[ind_two])
        cos_distances.append(embeddings_dist(group_one, group_two))
        print(f"Finished trial {i}")
    
    count = 0
    for d in cos_distances:
        if d >= control_dist:
            count += 1
    p_value = count / 10000
    print(f"P-value={p_value}")

def main():
    bootstrap()
if __name__ == '__main__':
    main()   
