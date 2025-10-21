import os
import openai
import random
import json
import csv
import numpy as np

client = openai.OpenAI(api_key="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")

def format_prompt(vignette, question, gender):
    subject = "she" if gender == "woman" else "he"
    possessive = "her" if gender == "woman" else "his"
    prompt = vignette + " " + question + \
             "\nFormat your response as <yes/no>, if <yes> then <high/low>, and then explanation."
    return prompt.replace("[gender]", gender).replace("[subject]", subject).replace("[possessive]", possessive)

def chat_prompter():
    prompts = []
    with open("pain_prompts.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append((row['Vignette'], row['Question']))

    results = []
    for trial in range(6,11):
        for i, (vignette, question) in enumerate(prompts):
            for gender in ["woman", "man"]:
                prompt_text = format_prompt(vignette, question, gender)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=0.7,
                        max_tokens=500
                    )
                    reply = response.choices[0].message.content
                    results.append({
                        "prompt_number": i + 1,
                        "trial": trial,
                        "gender": gender,
                        "response": reply
                    })
                except Exception as e:
                    print(f"Error on prompt {i + 1}, trial {trial + 1}, gender {gender}: {e}")

            print(f"Finished prompt {i + 1}, trial {trial + 1}")

    with open("paintwo_sampled_responses.json", "w") as f:
        json.dump(results, f, indent=2)

def main():
    chat_prompter()

if __name__ == '__main__':
    main()