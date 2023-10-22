import json

def flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def clean_nested_lists(input_dict):
    cleaned_dict = {}

    for key, value in input_dict.items():
        cleaned_dict[key] = [flatten_list(value[0]), flatten_list(value[1])]

    return cleaned_dict

with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Create_ID_convertor\Reaction_Map_Uniprot_Pubchem.json', 'r') as file:
    data = json.load(file)

d = clean_nested_lists(data)

c = []
e = []
for key, value in d.items():

    c += value[0]
    e += value[1]

c = list(set(c))
e = list(set(e))
print(len(c), len(e))

with open('compound_list.json', 'w') as output_file:
    json.dump(c, output_file, indent=2)

with open('enzyme_list.json', 'w') as output_file:
    json.dump(e, output_file, indent=2)

with open('Reaction_Map_Uniprot_Pubchem_formatted.json', 'w') as output_file:
    json.dump(d, output_file, indent=2)
