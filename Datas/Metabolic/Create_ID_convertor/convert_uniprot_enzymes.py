import requests
import json
from bs4 import BeautifulSoup

enzyme_map = {}

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

t=0
for i in enzyme_list:

    web_scrape(i)
    t+=1
    print(100* t/len(enzyme_list))

output_file = r"C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Create_ID_convertor\enzymes_map.json"

# Write the dictionary to the JSON file
with open(output_file, 'w') as json_file:
    json.dump(enzyme_map, json_file, indent=4)  # indent for pretty formatting
  
