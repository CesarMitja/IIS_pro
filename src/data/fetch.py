from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

# Set up Selenium WebDriver
options = Options()
options.add_argument('--headless')  # Runs Chrome in headless mode.
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=options)

def scrape_data():
    url = "https://www.nepremicnine.net/24ur/oglasi-prodaja/slovenija/"
    driver.get(url)
    time.sleep(5)  # Wait for the page to load

    properties = driver.find_elements_by_css_selector('.property-box')

    results = []
    for property in properties:
        try:
            location = property.find_element_by_css_selector('h2.p-0.m-0').text
            details = property.find_element_by_css_selector('p[itemprop="description"]').text.split(',')
            size = details[0].strip()
            year_built = details[1].split('zgrajeno l.')[1].strip()
            num_rooms = details[0].split(' ')[1]
            price = property.find_element_by_css_selector('h6').text.strip()
            city = location

            results.append({
                'location': location,
                'size': size,
                'year_built': year_built,
                'num_rooms': num_rooms,
                'price': price,
                'city': city
            })
        except Exception as e:
            print(f"Error parsing property details: {e}")

    return results

if __name__ == "__main__":
    data = scrape_data()
    print(data)

driver.quit()
