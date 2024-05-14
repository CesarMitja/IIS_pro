from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
import csv
import os

# Nastavitve za Selenium WebDriver
options = Options()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1200")

# Za debug: onemogočite headless način
options.headless = False

# Service z uporabo webdriver-manager
service = Service(ChromeDriverManager().install())

# Inicializacija Chrome Driverja
driver = webdriver.Chrome(service=service, options=options)

def scrape_nepremicnine():
    # URL ciljne strani
    url = 'https://www.nepremicnine.net/24ur/oglasi-prodaja/slovenija/'
    driver.get(url)
    
    print("URL Loaded: ", url)
    
    # Čakajte, da se stran naloži
    time.sleep(10)

    # Pridobivanje vseh oglasov na strani
    properties = driver.find_elements(By.CSS_SELECTOR, '.property-box')
    
    print("Number of properties found: ", len(properties))

    all_data = []

    for prop in properties:
        data = {
            'cena': '',
            'velikost': '',
            'leto_izgradnje': '',
            'st_sob': '',
            'mesto': ''
        }

        try:
            title_element = prop.find_element(By.CSS_SELECTOR, 'h2.p-0.m-0')
            data['mesto'] = title_element.text.strip()
            print("Mesto: ", data['mesto'])

            details = prop.find_element(By.CSS_SELECTOR, 'p[itemprop="description"]')
            details_text = details.text.strip()
            print("Details: ", details_text)

            price_element = prop.find_element(By.CSS_SELECTOR, 'h6')
            data['cena'] = price_element.text.strip().replace(' €', '').replace('.', '').replace(',', '.')
            print("Cena: ", data['cena'])

            size_match = re.search(r'(\d+,\d+|\d+) m2', details_text)
            if size_match:
                data['velikost'] = size_match.group(0).replace(' m2', '').replace(',', '.')
            print("Velikost: ", data['velikost'])

            year_match = re.search(r'zgrajeno l. (\d{4})', details_text)
            if year_match:
                data['leto_izgradnje'] = year_match.group(1)
            print("Leto izgradnje: ", data['leto_izgradnje'])

            rooms_match = re.search(r'(\d+)-sobno', details_text)
            if rooms_match:
                data['st_sob'] = rooms_match.group(1)
            print("Število sob: ", data['st_sob'])

        except Exception as e:
            print(f"An error occurred while parsing property: {e}")

        all_data.append(data)

    return all_data

# Shranjevanje podatkov v CSV datoteko
def save_to_csv(data, filename='nepremicnine_data.csv'):
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
            print(f"Saved: {item}")

# Pridobivanje podatkov in shranjevanje
data = scrape_nepremicnine()
save_to_csv(data)

# Zapiranje brskalnika
driver.quit()