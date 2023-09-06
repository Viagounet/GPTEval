from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("mteb/summeval")

# Convert the dataset to a pandas DataFrame
data = {
    "machine": dataset["test"]["machine_summaries"],
    "human": dataset["test"]["human_summaries"],
    "relevance": dataset["test"]["relevance"],
    "coherence": dataset["test"]["coherence"],
    "fluency": dataset["test"]["fluency"],
    "consistency": dataset["test"]["consistency"],
    "text": dataset["test"]["text"],
    "id": dataset["test"]["id"],
}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_filename = "summeval_dataset.csv"
df.to_csv(csv_filename, index=False, sep="\t")

print(f"Dataset saved to {csv_filename}")
