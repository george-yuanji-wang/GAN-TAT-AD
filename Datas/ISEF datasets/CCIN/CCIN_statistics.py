import csv

csv_file = r"C:\Users\George\Desktop\ISEF datasets\CCIN\chemical_chemical.links.detailed.v5.0.csv"
summary_file = r"C:\Users\George\Desktop\ISEF datasets\CCIN\chemical_chemical.links.detailed.v5.0.summary.txt"

# Read the CSV file
data = []
with open(csv_file, "r") as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header row
    for row in reader:
        data.append(row)

# Calculate summary statistics
num_rows = len(data)
num_columns = len(header)
column_types = ["string"] * num_columns
min_scores = [float('inf')] * num_columns
max_scores = [float('-inf')] * num_columns
avg_scores = [0] * num_columns
print(1)
for row in data: 
    for i, score in enumerate(row):
        if i == 0 or i == 1:  # Handle first and second columns as strings
            continue
        score = float(score)
        min_scores[i] = min(min_scores[i], score)
        max_scores[i] = max(max_scores[i], score)
        avg_scores[i] += score

# Calculate average scores
avg_scores = [round(total / num_rows, 2) for total in avg_scores]
print(1)
# Write the summary to a text file
with open(summary_file, "w") as file:
    file.write(f"Number of rows: {num_rows}\n")
    file.write(f"Number of columns: {num_columns}\n")
    file.write("\nSummary statistics:\n")
    file.write("Column\tType\tMinimum\tMaximum\tAverage\n")
    for i, column in enumerate(header):
        file.write(f"{column}\t{column_types[i]}\t{min_scores[i]}\t{max_scores[i]}\t{avg_scores[i]}\n")

print("Summary file created successfully.")