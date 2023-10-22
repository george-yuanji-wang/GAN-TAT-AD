import requests
from bs4 import BeautifulSoup
import re
import json
from tqdm import tqdm

Reaction_dict = {}

def extract_metabolic_pathway(url):
    global Reaction_dict
    # Send an HTTP request to the URL.
    response = requests.get(url)
    Smp = {}
    # Check if the request was successful (status code 200).
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup.
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the <div class="definition"> element.
        definition_div = soup.find('div', class_='definition')

        # Check if the definition_div is found.
        if definition_div:
            # Navigate to <table>, <tbody>, and find the relevant <tr> element.
            #print("definition_div found")
            table = definition_div.find('table')
            if table:
                #print("table found")
                
                relevant_trs = table.find_all('tr')

                # Iterate through each <tr> element and check for specified <td> elements.
                html_extract_data_store = []
                for tr in relevant_trs:
                    tds = tr.find_all('td', string=['Entry', 'Name', 'Reaction'])
                    # Check if the <tr> contains the specified <td> elements.
                    if any(tds):
                        # Extract information from the <tr> element.
                        for td in tr.find_all('td'):
                            if td.text not in ['Entry', 'Name', 'Reaction']:
                                html_extract_data_store.append(td.text)

                
                if len(html_extract_data_store) == 3:
                    Smp["Entry"] = html_extract_data_store[0].strip().replace('\n', '')
                    Smp["Name"] = html_extract_data_store[1]
                    reactions = html_extract_data_store[2]
                
                    formatted_reactions = re.sub(r'(C\d+)(R\d+)', r'\1\n\2', reactions)
    
                    formatted_reactions = formatted_reactions.split('\n')
                    formatted_reactions = [reaction.strip() for reaction in formatted_reactions]
                    formatted_reactions = formatted_reactions[1:]
            
                    formatted_reactions = [(line.split(' ', 1)[0], line.split(' ', 1)[1]) for line in formatted_reactions]
                        
                    Smp["Reactions"] = formatted_reactions
                    
                    for r in formatted_reactions:
                        if r[0] not in Reaction_dict.keys():
                            Reaction_dict[r[0]] = set(r[1])
                        else:
                            Reaction_dict[r[0]] = list(Reaction_dict[r[0]])
                            Reaction_dict[r[0]].append(r[1])
                            Reaction_dict[r[0]] = set(Reaction_dict[r[0]])
                else:
                    return (0, html_extract_data_store[0] + "error")

            else:
                print("No <table> found.")
        else:
            print("No <div class='definition'> found.")
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")

    if len(Smp) > 0:
        return Smp


modules = ['M00001', 'M00002', 'M00003', 'M00307', 'M00009', 'M00010', 'M00011', 'M00004', 'M00006', 'M00007', 'M00005', 'M00014', 'M00632', 'M00854', 'M00855', 'M00549', 'M00554', 'M00892', 'M00741', 'M00130', 'M00131', 'M00132', 'M00082', 'M00083', 'M00873', 'M00085', 'M00415', 'M00086', 'M00087', 'M00861', 'M00101', 'M00103', 'M00104', 'M00106', 'M00862', 'M00107', 'M00108', 'M00109', 'M00110', 'M00089', 'M00098', 'M00090', 'M00091', 'M00092', 'M00094', 'M00066', 'M00067', 'M00099', 'M00100', 'M00048', 'M00049', 'M00050', 'M00053', 'M00958', 'M00959', 'M00051', 'M00052', 'M00938', 'M00046', 'M00020', 'M00621', 'M00555', 'M00338', 'M00034', 'M00035', 'M00036', 'M00032', 'M00844', 'M00029', 'M00015', 'M00970', 'M00972', 'M00047', 'M00133', 'M00134', 'M00135', 'M00045', 'M00042', 'M00043', 'M00044', 'M00037', 'M00038', 'M00027', 'M00118', 'M00055', 'M00073', 'M00075', 'M00056', 'M00872', 'M00065', 'M00070', 'M00071', 'M00068', 'M00069', 'M00057', 'M00058', 'M00059', 'M00076', 'M00077', 'M00078', 'M00079', 'M00912', 'M00120', 'M00882', 'M00883', 'M00842', 'M00880', 'M00141', 'M00868', 'M00128', 'M00095', 'M00367']

extract_metabolic_pathway("https://www.genome.jp/module/M00009")

modules_extracted_list = []
for i in tqdm(modules, desc="Scraping Modules"):
    url_to_scrape = 'https://www.genome.jp/module/' + i
    test = extract_metabolic_pathway(url_to_scrape)
    if len(test) < 3:
        if test[0] == 0:
            print(test[1])
    else:
        modules_extracted_list.append(test)

test = extract_metabolic_pathway(url_to_scrape)
if len(test) < 3:
    if test[0] == 0:
        print(test[1])
else:
    modules_extracted_list.append(test)
test = extract_metabolic_pathway(url_to_scrape)
if len(test) < 3:
    if test[0] == 0:
        print(test[1])
else:
    modules_extracted_list.append(test)

with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Extract relevant reactions\modules_extracted_list.json', 'w') as json_file:
    json.dump(modules_extracted_list, json_file)

with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Extract relevant reactions\Reaction_dict.json', 'w') as json_file:
    json.dump(Reaction_dict, json_file)

