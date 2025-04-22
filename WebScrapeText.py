import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import numpy as np
import nltk
nltk.download('punkt_tab')  # Only needed once

def scrape_website(url, depth=0, max_depth=1, addl_els=None): #depth is to tell the scraper how many links it should follow after the first. Option to provide additional elements to scrape, like "li"
    try:
        headers = {
            'User-Agent': 'jkberry/1.0 (jkendall.berry@gmail.com)'
        }
        response = requests.get(url, headers=headers, timeout=10) #header is sent to the server for their info, timeout is how long to wait for the server to respond
        if response.status_code == 200: #status code 200 means all good, 404 is not found
            soup = BeautifulSoup(response.content, 'html.parser') #use .content instead of .text, better for character encoding. html.parser tells it that the response code will be html
            #BeautifulSoup helps us sort through the html response we get from the server
            # Extract title
            title = soup.title.string if soup.title else "No title found"
            
            # Extract main content
            content = ""
            main_content = soup.find_all(['p']) #these are different labels in the html code, for paragraph, headers, and list item
            for element in main_content:
                content += element.get_text() + " "
            
            # Extract links for further crawling if needed
            links = []
            if depth < max_depth: #this is set outside the function
                base_domain = urlparse(url).netloc #this is the network location, or the base domain, as the variable is named
                for link in soup.find_all('a', href=True): #find_all is a BeautifulSoup function that returns a list, a denotes a link in html, and href is the link url
                    href = link['href']
                    if href.startswith('http') and base_domain in href: #the base_domain part means this doesn't run if the base domain was None, and it does run if base_domain is defined
                        links.append(href)
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'links': links[:5]  # Limit to 5 links to prevent excessive crawling
            }
        else:
            print(f"Failed to retrieve {url}: Status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def split_sent(text):
    sentences = nltk.sent_tokenize(text)
    return sentences