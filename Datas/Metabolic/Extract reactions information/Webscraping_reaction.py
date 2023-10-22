import json
import requests
from bs4 import BeautifulSoup
import re
import time
from alive_progress import alive_bar

# Specify the path to your JSON file
json_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Extract reactions information\reactions_list.json'

# Initialize a list to store Reaction values
reaction_list = []
unsuccesful_url = []

# Read data from the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

    # Iterate through each dictionary in the list
    for entry in data:
        # Check if the dictionary has a key called "Reaction"
        if 'Reaction' in entry:
            # Add the "Reaction" value to the list
            reaction_list.append(entry['Reaction'])

Reaction_nodes = {}
# Specify the URL

def web_scrap(url):
    
    global Reaction_nodes
    global unsuccesful_url

    enzymes = []
    # Make a GET request to the website
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200 or response.status_code == 403:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the <td> element with class "fr2 w1"
        target_td = soup.find('td', class_='fr2 w1')

        if target_td:
            # Extract information from the first, fourth, and eighth <tr> elements
            tr_elements = target_td.find_all('tr')
            
            if len(tr_elements) >= 8:

                # Extract information for Entry
                a = target_td.find('td', class_="tal pd0")
                entry_span = a.find('span', class_='nowrap')
                entry_text = entry_span.text.strip() if entry_span else ''

                fr2_w1_td = soup.find('td', class_='fr2 w1')

                # Define the pattern to match in the href attribute
                pattern = re.compile(r'/entry/C\d+')

                # Find all <a> elements with the specified pattern inside <td class="fr2 w1">
                equation = fr2_w1_td.find_all('a', href=pattern)
                formula = []
                for i in equation:
                    formula.append(i.text.split()[0])

                '''# Extract information for Enzyme
                pattern = re.compile(r'/entry/\d+\.\d+\.\d+\.\d+')
                enzyme = target_td.find_all('a', href=pattern)
                enzymes = []
                for i in enzyme:
                    enzymes.append(i.text.split()[0])
                enzymes = set(enzymes)
                enzymes = list(enzymes)'''

                for tr in tr_elements:
    
                    span_element = tr.find('span', class_='nowrap') 

                    if span_element:
                        if span_element.text.strip() == "Enzyme":

                            # If <th> with <span> is found, find the corresponding <td class="td20 defd"> in the same <tr>
                            cel = tr.find('div', class_='cel')

                            if cel:
                
                                # Find the <a> element with an href attribute of the form "/entry/x.x.x.x"
                                href_pattern = re.compile(r'/entry/\d+\.\d+\.\d+\.\d+')
                                enzyme = cel.find_all('a', href=href_pattern)
                                
                                for i in enzyme:
                                    enzymes.append(i.text.split()[0])
                                enzymes = set(enzymes)
                                enzymes = list(enzymes)

                reaction = entry_text.split()[0]

                Reaction_nodes[reaction] = (formula, enzymes)

            else:
                print('Not enough <tr> elements under <td class="fr2 w1">')
        else:
            print('No <td> element with class "fr2 w1" found')
            unsuccesful_url.append(url)
            print(url)
    else:
        print(f'Failed to retrieve the page. Status code: {response.status_code}')\
        
    time.sleep(0.1)


with alive_bar(len(reaction_list), title="Scraping Modules") as bar:
    for reaction_url in reaction_list:
        url = "https://www.kegg.jp/entry/" + reaction_url
        web_scrap(url)
        bar()  # Increment the progress bar

for i in unsuccesful_url:
    print(i)
    web_scrap(i)


file_path = "Reaction_Map.json"

# Write the dictionary to the JSON file
with open(file_path, "w") as file:
    json.dump(Reaction_nodes, file, indent=2)
print(f'Dictionary has been successfully converted to {file_path}')

print(unsuccesful_url)