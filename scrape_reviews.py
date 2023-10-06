# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:32:05 2023

@author: rb
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time

def scrape_reviews(url):
    driver = webdriver.Firefox()
    driver.get(url)
    #page_url = f'{url}?page={page_number}'
    #response = requests.get(url)
    # if response.status_code != 200:
    #     raise Exception(f'Failed to retrieve the page: {response.status_code}')
    # soup = BeautifulSoup(response.text, 'html.parser')
    reviews = []
    count = 0
    
    while True:
        try:
            load_more_button = driver.find_element(By.CLASS_NAME, 'load-more-data')  # Update the CSS selector
            load_more_button.click()
            time.sleep(2)  # Wait for the new content to load
            print('loading more content...')
            count = count + 1
            if count == 50:
                break
        except NoSuchElementException:
            break  # No more "Load More" button found
    review_divs = driver.find_elements(By.CSS_SELECTOR, 'div.lister-item') # Update the CSS selector
    for review_div in review_divs:
        try:
            title = (review_div.find_element(By.CLASS_NAME, 'title').get_attribute('text'))
            body = (review_div.find_element(By.CSS_SELECTOR, 'div.text').get_attribute('innerHTML'))
            rating_parent = review_div.find_element(By.CLASS_NAME, 'rating-other-user-rating')
            rating = (rating_parent.find_element(By.TAG_NAME, 'span').text)
        except NoSuchElementException:
            print(f'Failed to extract rating for review: {review_div.get_attribute("outerHTML")}')
            rating = None
            continue  # Skip this review and move to the next one
        
        # Only append the review if a rating was found
        if rating is not None:
            reviews.append({'title': title, 'body': body, 'rating': rating})

        print(rating)
        # reviews.append({'title': title, 'body': body, 'rating': rating})

    driver.quit()
    return reviews
    # for review_div in soup.find_all('div', class_='review-container'):
    #     title = review_div.find('a', class_='title').text
    #     # rating_element = review_div.find('span', class_='rating')
    #     # if rating_element is None:
    #     #     continue
    #     body = review_div.find('div', class_='text').text
    #     rating = review_div.find('span', class_='').text
    #     # rating_class = rating_element['class']
    #     # rating = rating_class[2].split('-')[1]
    #     reviews.append({'title' : title, 'body': body, 'rating': rating})
    # return reviews

if __name__ == "__main__":
    url = 'https://www.imdb.com/title/tt1517268/reviews?ref_=tt_urv'
    # all_reviews = []
    # for page_number in range(1, 60):  # Pages 1 through 5
    #     print(f'Scraping page {page_number}...')
    #     reviews = scrape_reviews(url, page_number)
    #     all_reviews.extend(reviews)
    all_reviews = scrape_reviews(url)
    df = pd.DataFrame(all_reviews)
    df.to_csv('reviews.csv', index=False)
