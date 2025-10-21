import openai
import csv
import json
import numpy as np

client = openai.OpenAI(api_key="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")  # Replace with your actual key

def extract_prob(token_probs, target_tokens):
    for tok in target_tokens:
        if tok in token_probs:
            return token_probs[tok]
    return None

def parse_token_probs(response):
    logprobs = response.choices[0].logprobs
    tokens = logprobs.tokens[len(logprobs.tokens) - len(logprobs.token_logprobs):]
    probs = [np.exp(lp) if lp is not None else None for lp in logprobs.token_logprobs]
    return dict(zip(tokens, probs))

def parse_decision_and_dosage(token_probs):
    decision = None
    if extract_prob(token_probs, ['Yes', 'Yes.']) is not None:
        decision = 'Yes'
    elif extract_prob(token_probs, ['No', 'No.']) is not None:
        decision = 'No'

    dosage = None
    if decision == 'Yes':
        if extract_prob(token_probs, ['High', 'High.']):
            dosage = 'High'
        elif extract_prob(token_probs, ['Low', 'Low.']):
            dosage = 'Low'

    return decision, dosage

def format_prompt(vignette, question, gender):
    subject = "she" if gender == "woman" else "he"
    possessive = "her" if gender == "woman" else "his"
    prompt = vignette + " " + question + \
             "\nFormat your response as <yes/no>, if <yes> then <high/low>, and then explanation.\nAnswer: "
    return prompt.replace("[gender]", gender).replace("[subject]", subject).replace("[possessive]", possessive)

def run_prompts(csv_file="pain_prompts.csv", output_file="pain_token_probs.json"):
    results = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        prompts = list(reader)

    for i, row in enumerate(prompts):
        for gender in ["woman", "man"]:
            prompt_text = format_prompt(row['Vignette'], row['Question'], gender)
            try:
                response = client.completions.create(
                    model="text-davinci-003",  # ✅ Switched model
                    prompt=prompt_text,
                    temperature=0,
                    max_tokens=5,
                    logprobs=5,
                    echo=True
                )
                response_text = response.choices[0].text.strip()
                token_probs = parse_token_probs(response)
                decision, dosage = parse_decision_and_dosage(token_probs)

                results.append({
                    "prompt_number": i + 1,
                    "gender": gender,
                    "response": response_text,
                    "decision": decision,
                    "dosage": dosage,
                    "token_probs": token_probs
                })
            except Exception as e:
                print(f"Error on prompt {i + 1}, gender {gender}: {e}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

def main():
    run_prompts()

if __name__ == '__main__':
    main()
