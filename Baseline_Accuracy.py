import pandas as pd

# Load the CSV files
correct_answers = pd.read_csv('/content/Q_A_ID_training.csv')
responses = pd.read_csv('/content/Trainingques_Output_notfinetuned.csv')

# Merge the two dataframes on Question_ID
merged_df = pd.merge(correct_answers, responses, on='Question_ID', suffixes=('_correct', '_response'))

# Calculate the number of correct answers
correct_count = (merged_df['Answer_ID_correct'] == merged_df['Answer_ID_response']).sum()

# Calculate the total number of answers
total_count = len(merged_df)

# Calculate accuracy
accuracy = correct_count / total_count

# Print the results
print(f'Number of correct answers: {correct_count}')
print(f'Total number of answers: {total_count}')
print(f'Baseline Accuracy: {accuracy:.2%}')
