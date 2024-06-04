import requests
import csv
from datetime import datetime, timedelta
import os 

def timestamp_to_date(timestamp):
    return datetime.fromtimestamp(int(timestamp) / 1000)

# Get yesterday's date
yesterday = datetime.now() - timedelta(1)
print("Yesterday's date:", yesterday.strftime('%Y-%m-%d'))

url = "https://zillow56.p.rapidapi.com/search"

querystring = {"location":"houston, tx","output":"json","status":"forRent"}

headers = {
	"x-rapidapi-key": "9e5246f4e9msh9943232bbba051ap10f36cjsncd7485e246c0",
	"x-rapidapi-host": "zillow56.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())

data = response.json()

# Check and process each listing in the results
new_listings = []
for listing in data['results']:
    # Extract the 'datePriceChanged' or use 'timeOnZillow' to determine the listing date
    date_price_changed = listing.get('datePriceChanged')

    if date_price_changed:
        listing_date = timestamp_to_date(date_price_changed)
        # Check if the listing date is yesterday or today
        if listing_date > yesterday:
            new_listings.append(listing)
    elif listing.get('daysOnZillow') == 0:
        new_listings.append(listing)

# Print out new listings from the last day
print(f"Found {len(new_listings)} new listings from the last day.")

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 
filename = os.path.join(project_root, 'data', 'raw', 'listings_rent.csv')
file_exists = os.path.isfile(filename)
# Open the file in write mode
with open(filename, mode='a', newline='', encoding='utf-8') as file:
    # Define the field names for the CSV
    fieldnames = [ 'Price', 'Bedrooms', 'Bathrooms', 'Living Area', 'Type']
    
    # Create a CSV DictWriter object
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    # Write the header
    if not file_exists:
        writer.writeheader()
    
    # Write data for each new listing
    for listing in new_listings:
        writer.writerow({
            'Price': listing.get('price'),
            'Bedrooms': listing.get('bedrooms'),
            'Bathrooms': listing.get('bathrooms'),
            'Living Area': listing.get('livingArea'),
            'Type': listing.get('homeType')
        })

print(f"Data saved to {filename}")

