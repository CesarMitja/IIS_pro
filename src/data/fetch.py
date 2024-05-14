from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import re
import csv
import os

# Nastavitev Selenium WebDriver
chromedriver_path = Service(executable_path='src/data/chromedriver.exe')
options = Options()
driver = webdriver.Chrome(service=chromedriver_path, options=options)

def scrape_nepremicnine():
    # URL ciljne strani
    url = 'https://www.nepremicnine.net/24ur/oglasi-prodaja/slovenija/'
    driver.get(url)
    time.sleep(5)  # Pauza, da se stran popolnoma naloži

    # Pridobivanje vseh oglasov na strani
    properties = driver.find_elements(By.CSS_SELECTOR, '.property-box')

    all_data = []

    for prop in properties:
        # Inicializacija podatkov za vsak oglas
        data = {
            'cena': '',
            'velikost': '',
            'leto_izgradnje': '',
            'st_sob': '',
            'mesto': ''
        }

        try:
            # Zajem naslova oglasa
            title_element = prop.find_element(By.CSS_SELECTOR, 'h2.p-0.m-0')
            data['mesto'] = title_element.text.strip()

            # Zajem podrobnosti oglasa
            details = prop.find_element(By.CSS_SELECTOR, 'p[itemprop="description"]')
            details_text = details.text.strip()

            # Zajem cenovnih podatkov
            price_element = prop.find_element(By.CSS_SELECTOR, 'h6')
            data['cena'] = price_element.text.strip().replace(' €', '').replace('.', '').replace(',', '.')

            # Pridobivanje velikosti, leta izgradnje in števila sob iz opisa
            size_match = re.search(r'(\d+,\d+|\d+) m2', details_text)
            if size_match:
                data['velikost'] = size_match.group(0).replace(' m2', '').replace(',', '.')

            year_match = re.search(r'zgrajeno l. (\d{4})', details_text)
            if year_match:
                data['leto_izgradnje'] = year_match.group(1)

            rooms_match = re.search(r'(\d+)-sobno', details_text)
            if rooms_match:
                data['st_sob'] = rooms_match.group(1)

        except Exception as e:
            print(f"An error occurred while parsing property: {e}")

        # Dodajanje podatkov o oglasu v seznam vseh podatkov
        all_data.append(data)

    return all_data

# Shranjevanje podatkov v CSV datoteko
def save_to_csv(data, filename='datanepremicnine_data.csv'):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 
    filename = os.path.join(project_root, 'data', 'raw', 'current_data.csv')
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        fieldnames = ['cena', 'velikost', 'leto_izgradnje', 'st_sob', 'mesto']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()  # Pišemo glavo samo, če datoteka še ne obstaja

        for item in data:
            writer.writerow(item)

# Pridobivanje podatkov in shranjevanje
data = scrape_nepremicnine()
save_to_csv(data)

# Zapiranje brskalnika
driver.quit()