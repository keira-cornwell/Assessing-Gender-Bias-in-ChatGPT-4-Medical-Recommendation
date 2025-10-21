import json
import pandas as pd
from scipy.stats import wilcoxon

# === Load your JSON file ===
with open("pain_sampled_responses.json", "r") as f:
    data = json.load(f)
with open("paintwo_sampled_responses.json", "r") as ftwo:
    data = json.load(ftwo)

# === Parse responses ===
records = []
for item in data:
    resp = item["response"].lower()
    first_words = resp.split()
    decision = None
    dosage = None

    if first_words:
        if "no" in first_words[0]:
            decision = "No"
        elif "yes" in first_words[0]:
            decision = "Yes"
            if len(first_words) > 1:
                if "low" in first_words[1]:
                    dosage = "Low"
                elif "high" in first_words[1]:
                    dosage = "High"

    records.append({
        "prompt_number": item["prompt_number"],
        "trial": item["trial"],
        "gender": item["gender"],
        "decision": decision,
        "dosage": dosage
    })

df = pd.DataFrame(records)

# === Compute empirical probabilities ===
summary = df.groupby(["prompt_number", "gender"]).agg(
    P_No=('decision', lambda x: (x == "No").mean()),
    P_Low=('dosage', lambda x: (x == "Low").mean())
).reset_index()

# === Pivot to compare woman vs. man ===
pivot = summary.pivot(index="prompt_number", columns="gender")
pivot.columns = [f"{var}_{gender}" for var, gender in pivot.columns]
pivot = pivot.dropna()  # Keep only rows with both genders present

# === Run Wilcoxon signed-rank tests ===
stat_no, p_no = wilcoxon(pivot["P_No_woman"], pivot["P_No_man"])
stat_low, p_low = wilcoxon(pivot["P_Low_woman"], pivot["P_Low_man"])

# === Print results ===
print("===== Wilcoxon Signed-Rank Test Results =====")
print(f"Treatment Denial (P_No): W={stat_no:.2f}, p={p_no:.4f}")
print(f"Low Dosage (P_Low):      W={stat_low:.2f}, p={p_low:.4f}")

# === Optional: Show full table for review ===
print("\n===== Paired Probability Table =====")
print(pivot[["P_No_woman", "P_No_man", "P_Low_woman", "P_Low_man"]])

avg_diff = (pivot["P_No_woman"] - pivot["P_No_man"]).mean()
print(f"On average, women were {avg_diff*100:.1f}% more likely to receive a 'No' than men.")
