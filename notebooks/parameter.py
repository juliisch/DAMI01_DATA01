"""
Digital Business University of Applied Sciences
Data Science und Management (M. Sc.)
DAMI01 / DATA01 Data Analytics
Prof. Dr. Daniel Ambach
Julia Schmid (200022)


In dieser Datei sind die Paramter definiert. 
"""

# Importe
import os

# Pfade zu den Daten 
BASE_PATH = os.path.dirname(os.getcwd())
INPUT_ORIGIN_PATH = os.path.join(BASE_PATH, "data/origin")
INPUT_PATH = os.path.join(BASE_PATH, "data/processed/")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")

# Quelle: Claude AI (Anthropic). (2026a). Antwort von ChatGPT auf eine Anfrage zu Erstellung einer Kategorien-Dictionary der Variable description. Anhang C.
map_description = {
    "Food_and_Beverage": [
        'Miscellaneous Food Stores', 'Grocery Stores, Supermarkets',
        'Fast Food Restaurants', 'Eating Places and Restaurants',
        'Drinking Places (Alcoholic Beverages)',
        'Package Stores, Beer, Wine, Liquor'
    ],
    "Retail_General": [
        'Department Stores', 'Discount Stores', 'Wholesale Clubs',
        'Gift, Card, Novelty Stores', 'Antique Shops', 'Floor Covering Stores',
        'Miscellaneous Home Furnishing Stores',
        'Furniture, Home Furnishings, and Equipment Stores',
        'Household Appliance Stores'
    ],
    "Clothing_and_Apparel": [
        'Family Clothing Stores', "Women's Ready-To-Wear Stores",
        'Sports Apparel, Riding Apparel Stores', 'Shoe Stores', 'Leather Goods'
    ],
    "Electronics_and_Technology": [
        'Electronics Stores', 'Computers, Computer Peripheral Equipment',
        'Semiconductors and Related Devices', 'Computer Network Services',
        'Digital Goods - Media, Books, Apps', 'Digital Goods - Games',
        'Cable, Satellite, and Other Pay Television Services',
        'Telecommunication Services'
    ],
    "Health_and_Medical": [
        'Medical Services', 'Hospitals', 'Doctors, Physicians', 'Dentists and Orthodontists',
        'Podiatrists', 'Chiropractors', 'Drug Stores and Pharmacies',
        'Optometrists, Optical Goods and Eyeglasses'
    ],
    "Beauty_and_Personal_Care": [
        'Beauty and Barber Shops', 'Cosmetic Stores', 'Laundry Services'
    ],
    "Automotive": [
        'Service Stations', 'Automotive Service Shops', 'Automotive Body Repair Shops',
        'Automotive Parts and Accessories Stores', 'Car Washes', 'Towing Services',
        'Taxicabs and Limousines'
    ],
    "Travel_and_Transportation": [
        'Airlines', 'Cruise Lines', 'Passenger Railways', 'Bus Lines',
        'Railroad Passenger Transport', 'Railroad Freight',
        'Motor Freight Carriers and Trucking',
        'Local and Suburban Commuter Transportation',
        'Travel Agencies', 'Tolls and Bridge Fees'
    ],
    "Entertainment_and_Recreation": [
        'Athletic Fields, Commercial Sports', 'Recreational Sports, Clubs',
        'Amusement Parks, Carnivals, Circuses', 'Motion Picture Theaters',
        'Theatrical Producers', 'Betting (including Lottery Tickets, Casinos)',
        'Sporting Goods Stores', 'Music Stores - Musical Instruments'
    ],
    "Home_and_Garden": [
        'Hardware Stores', 'Lumber and Building Materials',
        'Brick, Stone, and Related Materials', 'Gardening Supplies',
        'Florists Supplies, Nursery Stock and Flowers',
        'Lawn and Garden Supply Stores', 'Lighting, Fixtures, Electrical Supplies',
        'Upholstery and Drapery Stores', 'Artist Supply Stores, Craft Shops'
    ],
    "Financial_and_Legal": [
        'Money Transfer', 'Insurance Sales, Underwriting',
        'Accounting, Auditing, and Bookkeeping Services',
        'Tax Preparation Services', 'Legal Services and Attorneys'
    ],
    "Professional_Services": [
        'Detective Agencies, Security Services', 'Cleaning and Maintenance Services',
        'Heating, Plumbing, Air Conditioning Contractors',
        'Postal Services - Government Only'
    ],
    "Books_and_Media": [
        'Book Stores', 'Books, Periodicals, Newspapers'
    ],
    "Lodging": [
        'Lodging - Hotels, Motels, Resorts'
    ],
    "Utilities": [
        'Utilities - Electric, Gas, Water, Sanitary'
    ],
    "Metals_and_Manufacturing": [
        'Non-Precious Metal Services', 'Ironwork', 'Non-Ferrous Metal Foundries',
        'Miscellaneous Machinery and Parts Manufacturing', 'Precious Stones and Metals',
        'Miscellaneous Metalwork', 'Welding Repair', 'Pottery and Ceramics',
        'Ship Chandlers', 'Tools, Parts, Supplies Manufacturing',
        'Electroplating, Plating, Polishing Services', 'Miscellaneous Metals',
        'Steel Products Manufacturing', 'Heat Treating Metal Services', 'Steelworks',
        'Fabricated Structural Metal Products',
        'Bolt, Nut, Screw, Rivet Manufacturing', 'Miscellaneous Metal Fabrication',
        'Steel Drums and Barrels', 'Coated and Laminated Products',
        'Miscellaneous Fabricated Metal Products',
        'Industrial Equipment and Supplies'
    ]
}

# Quelle: Claude AI (Anthropic). (2026b). Antwort von ChatGPT auf eine Anfrage zu Erstellung einer Kategorien-Dictionary der Variable merchant_state. Anhang B.
map_merchant_state = {
    "US_States": [
        'ND', 'IA', 'CA', 'IN', 'MD', 'NY', 'TX', 'HI', 'PA', 'WI', 'GA', 'AL',
        'CT', 'WA', 'MA', 'CO', 'NJ', 'OK', 'MT', 'FL', 'AZ', 'KY', 'LA', 'IL',
        'OH', 'MO', 'MI', 'KS', 'NC', 'AR', 'TN', 'NM', 'SC', 'MN', 'NV', 'OR',
        'VA', 'SD', 'WV', 'ME', 'MS', 'RI', 'NH', 'DE', 'VT', 'ID', 'NE', 'DC',
        'UT', 'WY', 'AK', 'AA'
    ],
    "North_America": [
        'Mexico', 'Canada', 'Costa Rica', 'Dominican Republic', 'Guatemala',
        'Honduras', 'Belize', 'Panama', 'Jamaica', 'The Bahamas', 'Barbados',
        'Aruba', 'Antigua and Barbuda', 'Saint Vincent and the Grenadines',
        'Trinidad and Tobago', 'Haiti'
    ],
    "South_America": [
        'Colombia', 'Brazil', 'Peru', 'Chile', 'Argentina', 'Uruguay',
        'Ecuador', 'Suriname'
    ],
    "Europe": [
        'Germany', 'United Kingdom', 'Estonia', 'Lithuania', 'Netherlands',
        'Greece', 'Ireland', 'France', 'Italy', 'Denmark', 'Belgium',
        'Switzerland', 'Portugal', 'Finland', 'Norway', 'Hungary', 'Israel',
        'Monaco', 'Romania', 'Russia', 'Austria', 'Spain', 'Moldova', 'Croatia',
        'Sweden', 'Andorra', 'Czech Republic', 'Macedonia', 'Turkey', 'Luxembourg',
        'Slovakia', 'Ukraine', 'Montenegro', 'Iceland', 'Slovenia', 'Latvia',
        'Poland', 'Georgia', 'Cyprus', 'Serbia', 'Kosovo', 'Bosnia and Herzegovina',
        'Malta', 'Albania', 'Vatican City'
    ],
    "Asia": [
        'China', 'Taiwan', 'United Arab Emirates', 'Japan', 'Vietnam',
        'Singapore', 'Thailand', 'Indonesia', 'Philippines', 'Saudi Arabia',
        'South Korea', 'India', 'Mongolia', 'Hong Kong', 'Bangladesh',
        'Uzbekistan', 'Brunei', 'Pakistan', 'Oman', 'Bahrain', 'Malaysia',
        'Myanmar (Burma)', 'Iran', 'Lebanon', 'Iraq', 'Jordan', 'Yemen',
        'Qatar', 'Maldives', 'Kyrgyzstan', 'Sri Lanka'
    ],
    "Africa": [
        'South Africa',  'Benin', 'Sierra Leone', 'Kenya', 'Eritrea',
        'Cameroon', 'Nigeria', 'Algeria', 'Niger', 'Egypt', 'Ghana',
        'South Sudan', 'Burkina Faso', 'Zimbabwe', 'Cabo Verde',
        'Equatorial Guinea', 'Ethiopia', 'Swaziland', 'Mozambique', 'Zambia',
        'Liberia', 'Mali', 'Senegal', 'Gabon', 'Tunisia', 'Seychelles',
        "Cote d'Ivoire", 'Guinea', 'Morocco'
    ],
    "Oceania": [
        'New Zealand', 'Australia', 'Tuvalu', 'Marshall Islands', 'Micronesia',
        'Tonga', 'Solomon Islands', 'Fiji', 'Nauru', 'Vanuatu', 'Samoa'
    ]
}

# Quelle: Claude AI (Anthropic). (2026c). Antwort von ChatGPT auf eine Anfrage zu Erstellung einer Kategorien-Dictionary der Variable error_category. Anhang D.
map_error_category = {
    "No_Error": [
        'None'
    ],
    "Single_Error": [
        'Technical Glitch', 'Bad Expiration', 'Bad Card Number',
        'Insufficient Balance', 'Bad PIN', 'Bad CVV', 'Bad Zipcode'
    ],
    "Multiple_Errors": [
        'Insufficient Balance,Technical Glitch', 'Bad PIN,Insufficient Balance',
        'Bad PIN,Technical Glitch', 'Bad Expiration,Technical Glitch',
        'Bad Card Number,Bad Expiration', 'Bad Card Number,Insufficient Balance',
        'Bad Expiration,Insufficient Balance', 'Bad Card Number,Bad CVV',
        'Bad CVV,Technical Glitch', 'Bad Expiration,Bad CVV',
        'Bad CVV,Insufficient Balance', 'Bad Card Number,Technical Glitch',
        'Bad Zipcode,Insufficient Balance', 'Bad Zipcode,Technical Glitch',
        'Bad Card Number,Bad Expiration,Insufficient Balance'
    ]
}