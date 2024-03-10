

path = r'C:\Users\George\Desktop\ISEF-2023\Outcome_data\DoGSiteScorer\abcergre.txt'



fix = 0
with open(path, 'r') as file:
    lines = file.readlines()

    # Count the pattern "}}}{" in the modified content
    print("\t\t}\n\t}\n}\n{")
    print("\t\t}\n\t},")

    modified_lines = []
    i = 0
    while i < len(lines):
        #print(lines[i])
        # Check if the current line is "{" and the next line is "}"
        if lines[i].strip() == "}" and i + 1 < len(lines) and lines[i + 1].strip() == "{":
            # Add "," to replace these two lines
            modified_lines.append(",\n")
            # Skip the next line as it is part of the replaced pattern
            i += 2
            fix += 1
        else:
            # Add the current line to modified lines if it doesn't match the pattern
            modified_lines.append(lines[i])
            i += 1

    print(fix)
    # Write the modified lines to the output file
    with open(r'C:\Users\George\Desktop\ISEF-2023\Outcome_data\DoGSiteScorer\fixed_DogSiteScorer_data_withALLPredicted.json', 'w') as file:
        file.writelines(modified_lines)

    print("Replacement complete.")
    

