import requests
import json
from bs4 import BeautifulSoup

enzyme_map = {}

# Load your JSON file
with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Extract reactions information\Reaction_Map.json', 'r') as file:
    data = json.load(file)

def web_scrape(enzyme):
    global enzyme_map
    try:
        url = "https://enzyme.expasy.org/EC/" + enzyme
        # Send an HTTP request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the <div> with class "main-elmt"
            main_div = soup.find('div', class_='main-elmt', style='justify-self:center;')

            # Check if the main_div is found
            if main_div:
                # Create a list to store information from <td> containing "HUMAN"
                td_info_list = []
                convert = []
                # Iterate through each <tr> under the main_div
                for tr in main_div.find_all('tr'):
                    # Check if any <td> in the current <tr> contains "HUMAN"
                    if any("HUMAN" in td.get_text(strip=True) for td in tr.find_all('td')):
                        # Iterate through each <td> in the current <tr>
                        for td in tr.find_all('td'):
                            # Check if "HUMAN" is in the text of the current <td>
                            if "HUMAN" in td.get_text(strip=True):
                                # Append the text of the <td> to the list
                                t = str(td.get_text(strip=True))
                                td_info_list.append(t)

                        all = td_info_list[1:]
                        for _ in all:
                            convert.append(_.split(",")[0])

                        convert = list(set(convert))

                if enzyme not in enzyme_map.keys():
                    enzyme_map[enzyme] = convert
                else:
                    enzyme_map[enzyme] += convert

            else:
                print("No <div> with class 'main-elmt' found.")

        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")


a = r"C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Create_ID_convertor\enzymes_list.json"

with open(a, 'r') as file:
    enzyme_list = json.load(file)


a = r"C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Create_ID_convertor\chemical_list.json"

with open(a, 'r') as file:
    chemical_list = json.load(file)

chem_map = {}

def scrape_pubchem_info(chem):
    global chem_map
    # Send an HTTP request to the URL
    url = "https://www.genome.jp/entry/" + chem
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all <a> elements with an 'href' attribute including 'pubchem'
        relevant_a_elements = soup.find_all('a', href=lambda value: value and 'pubchem' in value.lower())

        # Initialize a list to store the information
        pubchem_info_list = []

        # Iterate through the relevant <a> elements and append their information to the list
        for a_element in relevant_a_elements:
            # Get text content of the <a> element
            a_text = a_element.get_text(strip=True)
            
            # Append the information to the list
            pubchem_info_list.append(a_text)

        if pubchem_info_list:
            chem_map[chem] = pubchem_info_list[0]
        else:
            print(chem)

    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None


# Function to replace values in the dictionary
def replace_values(original_dict, enzyme, compound):
    # Load mapping dictionaries from JSON files

    # Go through every "value" of the dictionary
    for key, value in original_dict.items():
        # Check if the value is in 'C' mapping
        new = []
        for i in range(len(value[0])):
            if value[0][i] in compound.keys():
                new.append(compound[value[0][i]])

        value[0] = new 

        # Check if the value is in 'X' mapping
        for i in range(len(value[1])):
            
            if value[1][i] in enzyme.keys():
   
                value[1][i] = enzyme[value[1][i]]

    return original_dict

t=0
for i in chemical_list:

    scrape_pubchem_info(i)
    t+=1
    print(100* t/len(chemical_list))

t=0
for i in enzyme_list:

    web_scrape(i)
    t+=1
    print(100* t/len(enzyme_list))

# Apply replacements to the data
replace_values(data, enzyme_map, chem_map)

# Save the modified data back to the JSON file
with open('Reaction_Map_Uniprot_Pubchem.json', 'w') as output_file:
    json.dump(data, output_file, indent=2)