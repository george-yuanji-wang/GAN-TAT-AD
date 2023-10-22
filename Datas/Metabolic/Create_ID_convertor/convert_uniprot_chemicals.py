import requests
import json
from bs4 import BeautifulSoup

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
    
t=0
for i in chemical_list:

    scrape_pubchem_info(i)
    t+=1
    print(100* t/len(chemical_list))

output_file = r"C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Create_ID_convertor\chemical_map.json"
# Write the dictionary to the JSON file
with open(output_file, 'w') as json_file:
    json.dump(chem_map, json_file, indent=4)  # indent for pretty formatting