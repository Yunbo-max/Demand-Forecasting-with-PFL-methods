# -*- coding = utf-8 -*-
# @time:18/03/2023 21:00
# Author:Yunbo Long
# @File:location_city.py
# @Software:PyCharm
import pandas as pd



text = '''Plymouth, Massachusetts, United States
Los Angeles, California, United States
San Juan, Puerto Rico, United States
Dallas, Texas, United States
Chicago, Illinois, United States
Nepalgunj, Bheri, Nepal
Honolulu, Hawaii, United States
Chapel Hill, North Carolina, United States
Sacramento, California, United States
Miami, Florida, United States
Azores, Portugal
Greeley, Colorado, United States
Philadelphia, Pennsylvania, United States
Cleveland, Ohio, United States
Albuquerque, New Mexico, United States
Mesa, Arizona, United States
Bellingham, Washington, United States
Memphis, Tennessee, United States
San Antonio, Texas, United States
Atlanta, Georgia, United States
Fargo, North Dakota, United States
Cape Town, Western Cape, South Africa
Bronx, New York, United States
Xingtai, Hebei, China
Bend, Oregon, United States
Salt Lake City, Utah, United States
Lincoln, Kansas, United States
New Orleans, Louisiana, United States
Alexandria, Virginia, United States
Middletown, Ohio, United States
Houston, Texas, United States
Las Vegas, Nevada, United States
McAllen, Texas, United States
Ann Arbor, Michigan, United States
Columbia, Missouri, United States
Jacksonville, Florida, United States
Rochester, New York, United States
El Cajon, California, United States
Minneapolis, Minnesota, United States
Amarillo, Texas, United States
El Paso, Texas, United States
Norfolk, Virginia, United States
Murfreesboro, Tennessee, United States
Quincy, Illinois, United States
San Jose, California, United States
Orlando, Florida, United States
Boise, Idaho, United States
Portland, Oregon, United States
Fresno, California, United States
Summerville, South Carolina, United States
Bismarck, North Dakota, United States
Oxnard, California, United States
Columbus, Ohio, United States
Riverside, California, United States
Dubuque, Iowa, United States
Greensburg, Pennsylvania, United States
Cancun, Quintana Roo, Mexico
St. Louis, Missouri, United States
Midland, Michigan, United States
Billings, Montana, United States
Tulsa, Oklahoma, United States
Tallahassee, Florida, United States
Raleigh, North Carolina, United States
Farmington Hills, Michigan, United States
Arlington, Virginia, United States
Laredo, Texas, United States
Elmira, New York, United States
Hartford, Connecticut, United States
Harrisburg, Pennsylvania, United States
Tucson, Arizona, United States
Tampa, Florida, United States
Rock Hill, South Carolina, United States
Niagara Falls, New York, United States
Louisville, Kentucky, United States
Rockford, Illinois, United States
Bayamon, Puerto Rico, United States
Bakersfield, California, United States
Kirkland, Washington, United States
El Centro, California, United States
Aurora, Colorado, United States
Roswell, New Mexico, United States
Overland Park, Kansas, United States
Boston, Massachusetts, United States
Eugene, Oregon, United States
Austin, Texas, United States
Eagle Pass, Texas, United States
Spokane, Washington, United States
Wichita, Kansas, United States
Opelousas, Louisiana, United States
Normal, Illinois, United States
Hickory, North Carolina, United States
Greenville, North Carolina, United States
Montrose, Colorado, United States
Baltimore, Maryland, United States
Sheboygan, Wisconsin, United States
Brooklyn, New York, United States
Farragut, Tennessee, United States
Rome, New York, United States
Indianapolis, Indiana, United States
Kentwood, Michigan, United States
San Francisco, California, United States
Irvine, California, United States
La Crosse, Wisconsin, United States
Reno, Nevada, United States
Columbia, South Carolina, United States
Merrillville, Indiana, United States
Lawton, Oklahoma, United States
Hagerstown, Maryland, United States
Henrico, Virginia, United States
Tulare, California, United States
Augusta, Georgia, United States
Tracy, California, United States
Pittsfield, Massachusetts, United States
Lompoc, California, United States
Jonesboro, Arkansas, United States
Yakima, Washington, United States
Springfield, Missouri, United States
Oceanside, California, United States
Edison, New Jersey, United States
Pueblo, Colorado, United States
Conway, Arkansas, United States
Gilroy, California, United States
College Station, Texas, United States
Inglewood, California, United States
Columbus, Georgia, United States
Smyrna, Georgia, United States
Naperville, Illinois, United States
Harlingen, Texas, United States
Lexington, Kentucky, United States
Unincorporated area, California, United States (approximation, no city nearby)
Salina, Kansas, United States
Aguadilla, Puerto Rico, United States
Del Rio, Texas, United States
Davenport, Iowa, United States
Washington, Pennsylvania, United States
Simi Valley, California, United States
Clarksville, Tennessee, United States
Klamath Falls, Oregon, United States
Clearfield, Utah, United States
Ontario, California, United States
Findlay, Ohio, United States
Corvallis, Oregon, United States
Tbilisi, Georgia (the country, not the U.S. state)
Mitchell, South Dakota, United States
Garland, Texas, United States
Napa, California, United States
Lancaster, California, United States
Huntington, New York, United States
Lancaster, Pennsylvania, United States
Birmingham, Alabama, United States
Baytown, Texas, United States
Susanville, California, United States
Turlock, California, United States
Brady, Texas, United States
Columbus, Ohio, United States
Elgin, Illinois, United States
Lansing, Michigan, United States
Cumberland, Maryland, United States
Edmond, Oklahoma, United States
Blacksburg, Virginia, United States
San Juan, Puerto Rico, United States
Cleveland, Ohio, United States
Wilmington, Delaware, United States
Bowling Green, Kentucky, United States
Catskill, New York, United States
Fort Worth, Texas, United States
Middletown, New York, United States
Bullhead City, Arizona, United States
Phoenix, Arizona, United States
Atlanta, Georgia, United States
Peoria, Illinois, United States
Milwaukee, Wisconsin, United States
Roseburg, Oregon, United States
Fort Lauderdale, Florida, United States
Danbury, Connecticut, United States
Melbourne, Florida, United States
Cincinnati, Ohio, United States
Terre Haute, Indiana, United States
Pinehurst, North Carolina, United States
Big Bear City, California, United States
Las Cruces, New Mexico, United States
Elyria, Ohio, United States
Meadville, Pennsylvania, United States
Mount Pleasant, Michigan, United States
Colorado Springs, Colorado, United States
Longmont, Colorado, United States
Venice, Veneto, Italy
Santa Cruz, California, United States
Yuma, Arizona, United States
Victorville, California, United States
Glenwood Springs, Colorado, United States
Caguas, Puerto Rico, United States
Folsom, California, United States
Ithaca, New York, United States
Fond du Lac, Wisconsin, United States
Woonsocket, Rhode Island, United States
Hendersonville, Tennessee, United States
Stow, Ohio, United States
Brownwood, Texas, United States
Wilkes-Barre, Pennsylvania, United States
Petoskey, Michigan, United States
Danville, Kentucky, United States
Vineland, New Jersey, United States
York, Pennsylvania, United States
Montclair, New Jersey, United States
Oak Lawn, Illinois, United States
Jacksonville, North Carolina, United States
Tyler, Texas, United States
San Marcos, Texas, United States
Canton, Ohio, United States
Hobbs, New Mexico, United States
Fontana, California, United States
Lakewood, New Jersey, United States
Athens, Ohio, United States
Jackson, Michigan, United States
Martinez, California, United States
San Mateo, California, United States
Santa Fe, New Mexico, United States
Santa Barbara, California, United States
El Paso, Texas, United States
Houston, Texas, United States
Brooklyn, New York, United States
Englewood, Ohio, United States
Astoria, Oregon, United States
San Luis Obispo, California, United States
Norristown, Pennsylvania, United States
Madison, Wisconsin, United States
Williamsport, Pennsylvania, United States
Annapolis, Maryland, United States
Corrubedo, Galicia, Spain
Wenatchee, Washington, United States
Dearborn, Michigan, United States
Turlock, California, United States
Anaheim, California, United States
San Diego, California, United States
Stockton, California, United States
Salem, Oregon, United States
Green Bay, Wisconsin, United States
Lafayette, Indiana, United States
San Juan, Puerto Rico, United States
Auburn, Washington, United States
Bowie, Maryland, United States
Philadelphia, Pennsylvania, United States
Dunkirk, New York, United States
Sparta, Tennessee, United States
Kendall, Florida, United States
Methuen, Massachusetts, United States
St. Louis, Missouri, United States
Fredericksburg, Virginia, United States
Gaithersburg, Maryland, United States
Atlanta, Georgia, United States
Waukegan, Illinois, United States
Gardena, California, United States
Salinas, California, United States
Muskegon, Michigan, United States
Mentor, Ohio, United States
Elyria, Ohio, United States
Newport News, Virginia, United States
Macon, Georgia, United States
Charlotte, North Carolina, United States'''

print(text)
# Splitting the text by lines and then by commas
rows = [line.split(', ') for line in text.strip().split('\n')]
print(rows)
columns = ['City', 'State', 'Country']

# Creating a DataFrame using Pandas
df = pd.DataFrame(rows)

# Saving the DataFrame to an Excel file
df.to_excel('location_names.xlsx', index=False)


