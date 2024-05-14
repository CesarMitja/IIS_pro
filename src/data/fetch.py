import requests
import csv
from datetime import datetime, timedelta
import os 

# Function to convert timestamp to datetime
def timestamp_to_date(timestamp):
    return datetime.fromtimestamp(int(timestamp) / 1000)

# Get yesterday's date
yesterday = datetime.now() - timedelta(1)
print("Yesterday's date:", yesterday.strftime('%Y-%m-%d'))

# Endpoint and headers
url = "https://zillow56.p.rapidapi.com/search"
querystring = {"location":"houston, tx"}
headers = {
    "X-RapidAPI-Key": "37525841acmshd3f8a8fdd884aabp1226a9jsnfc8f85254ca0",
    "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
}

# Send the request to the API
response = requests.get(url, headers=headers, params=querystring)
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
filename = os.path.join(project_root, 'data', 'raw', 'listings.csv')

# Open the file in write mode
with open(filename, mode='w', newline='', encoding='utf-8') as file:
    # Define the field names for the CSV
    fieldnames = ['Address', 'City', 'State', 'Zipcode', 'Price', 'Bedrooms', 'Bathrooms', 'Living Area', 'Lot Area', 'Type', 'Image', 'Date Listed']
    
    # Create a CSV DictWriter object
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()
    
    # Write data for each new listing
    for listing in new_listings:
        writer.writerow({
            'Address': listing.get('streetAddress'),
            'City': listing.get('city'),
            'State': listing.get('state'),
            'Zipcode': listing.get('zipcode'),
            'Price': listing.get('price'),
            'Bedrooms': listing.get('bedrooms'),
            'Bathrooms': listing.get('bathrooms'),
            'Living Area': listing.get('livingArea'),
            'Lot Area': listing.get('lotAreaValue'),
            'Type': listing.get('homeType'),
            'Image': listing.get('imgSrc'),
            'Date Listed': timestamp_to_date(listing['datePriceChanged']) if listing.get('datePriceChanged') else 'Today'
        })

print(f"Data saved to {filename}")