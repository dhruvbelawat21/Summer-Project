import json
import pandas as pd
from transformers import AutoTokenizer

# Load the dataset
with open('TeleQnA_training.txt', 'r') as f:
    questions = json.load(f)

answers = pd.read_csv('Q_A_ID_training.csv')

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
tokenizer.pad_token = tokenizer.eos_token

# Prepare the dataset
def prepare_data(questions, answers):
    data = []
    for idx, row in answers.iterrows():
        q_id = 'question ' + str(int(row['Question_ID'])) # Modify the key to match the format in the questions dictionary
        answer_id = row['Answer_ID']
 # Handle potential missing keys
        if q_id in questions:
            question_data = questions[q_id]
            prompt = f"""
            {question_data.get('question', '')} # Use .get() to handle missing keys
            "option 1": "{question_data.get('option 1', '')}",
            "option 2": "{question_data.get('option 2', '')}",
            "option 3": "{question_data.get('option 3', '')}",
            "option 4": "{question_data.get('option 4', '')}",
            Answer with the option number only: 1, 2, 3, or 4.
            """
            answer = f"{answer_id}"
            data.append((prompt, answer))
        else:
            print(f"Warning: Question ID {q_id} not found in questions dictionary.") # Print a warning for missing questions
    return data

# Generate the training data
training_data = prepare_data(questions, answers)
# Tokenize the dataset
def tokenize_data(data, tokenizer):
    inputs = []
    labels = []
    for prompt, answer in data:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        labels_ids = tokenizer(answer, return_tensors="pt")["input_ids"]
        inputs.append(input_ids)
        labels.append(labels_ids)
    return inputs, labels

inputs, labels = tokenize_data(training_data, tokenizer)
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer

# Load the model and tokenizer
model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
 tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare the dataset
class TelecomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'input_ids': self.inputs[idx].squeeze(), 'labels': self.labels[idx].squeeze()}

# Example data (replace with your actual data)
inputs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]  
labels = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]  

train_dataset = TelecomDataset(inputs, labels)

# Data collator for padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()
model.save_pretrained("./summerprof_fine_tuned_phi2")
tokenizer.save_pretrained("./summerproj_fine_tuned_phi2")
