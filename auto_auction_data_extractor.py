#!/usr/bin/env python
# coding: utf-8

# This script extracts tables from salvageautosauction website and stores them to the csv file

# In[9]:


from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from dateutil import parser
import pandas as pd


# In[10]:


def click_option(id, value):
    el = driver.find_element_by_id(id)
    for option in el.find_elements_by_tag_name('option'):
        if option.text == value:
            option.click()


# In[11]:


def extract_data(driver, make, from_year, to_year, resp_array, page_num):
    
    resp = []
    
    if page_num == 1:
        click_option('cboFrYear', from_year)
        click_option('cboToYear', to_year)
        click_option('cboMake', make)

        search = driver.find_element_by_name('btnSubmit')
        search.click()
        
        resp_array += extract_table_from_page(driver)
        
    while(True):
        
        page_num += 1
    
        next_link = "https://www.salvageautosauction.com/price_history/" + str(page_num)
        try:
            next_page = driver.find_element_by_xpath('//a[contains(@href, "%s")]' % next_link)
        except Exception as e:
            print(e)
            print('Final page number', page_num)
            return resp_array
        
        try:
            next_page.click()
        except Exception as e:
            print(e)
            print('Final page number', page_num)
            return resp_array

        try:
            resp = extract_table_from_page(driver)
        except Exception as e:
            print(e)
            print('Final page number', page_num)
            return resp_array
        
        resp_array += resp
                            
    return resp_array


# In[12]:


def extract_table_from_page(driver):
    x = driver.find_element_by_class_name('price_history')
    obj_array = []
    for idx, row in enumerate(x.find_elements_by_class_name('row')):
        if idx == 0:
            continue
        obj = {}
        items = row.text.split('  ')

        items = row.find_elements_by_css_selector(".col-xs-12.col-sm-6.col-md-4.col-lg-3")
        price_item = row.find_element_by_css_selector(".col-xs-12.col-sm-6.col-md-4.col-lg-2.last")

        if items[0].text and items[0].text != '  ':
            y = items[0].text.split('\n')
            model = y[0].lstrip().split(' ')
            model_year = model[0]
            model_name = ' '.join(model[1:])
            obj['Model Year'] = int(model_year)
            obj['Model'] = model_name
            obj['Title State/Type'] = y[1].split(':')[1].lstrip()
            obj['Location'] = y[2].split(':')[1].lstrip()
        if items[1].text and items[1].text != '  ':
            date = items[1].text.split('\n')[0].lstrip()
            obj['Auction Date'] = parser.parse(date)
        if items[2].text and items[2].text != '  ':
            actual_value = items[2].text.split('\n')[0].lstrip().split(' ')
            obj['Auctual Cash Value'] = float(actual_value[1].replace(',', ''))
            obj['Actual Cash Value Currency'] = actual_value[2] + ' ' + actual_value[0]
        if items[3].text and items[3].text != '  ':
            repair_cost = items[3].text.split('\n')[0].lstrip().split(' ')
            obj['Repair Cost'] = float(repair_cost[1].replace(',', ''))
            obj['Repair Cost Currency'] = repair_cost[2] + ' ' + repair_cost[0]
        if items[4].text and items[4].text != '  ':
            obj['Odometer'] = items[4].text.split('\n')[0].lstrip()
        if items[5].text and items[5].text != '  ':
            obj['Prim Damage'] = items[5].text.split('\n')[0].lstrip()
        if items[6].text and items[6].text != '  ':
            obj['Sec Damage'] = items[6].text.split('\n')[0].lstrip()
        if price_item.text and price_item.text != '  ':
            price_sold = price_item.text.split('\n')[0].lstrip().split(' ')
            obj['Price Sold or Highest Bid'] = float(price_sold[1].replace(',', ''))
            obj['Price Sold or Highest Bid Currency'] = price_sold[2] + ' ' + price_sold[0]
        if obj:
            obj_array.append(obj)
            
    return obj_array


# In[20]:


def query_and_save_data(driver, make, from_year, to_year, resp_array, page_num):
    resp = extract_data(driver, make, from_year, to_year, resp_array, page_num)

    df = pd.DataFrame(resp)
    file = 'cars_datasets\\{0}_from_{1}_to_{2}.csv'.format(make, from_year, to_year)
    df.to_csv(file, index=False)
    return resp


# In[24]:


from_year = '2020'
to_year = '2020'
make = 'Toyota'
options = Options()

# edit per user 
options.binary_location = "c:\\users\\mrkmd\\appdata\\local\\google\\chrome sxs\\application\\chrome.exe"
driver = webdriver.Chrome(options=options, executable_path="c:\\programdata\\chromedriver\\chromedriver.exe")
# edit per user 

driver.get('https://www.salvageautosauction.com/price_history')
resp = query_and_save_data(driver, make, from_year, to_year, [], 1)


# In case query_and_save_data crashes, go to the previous page on the browser and run the following line with page_num equals to the previous page number

# In[22]:


resp = query_and_save_data(driver, make, from_year, to_year, resp, 2641)
df = pd.DataFrame(resp)
file = 'cars_datasets\\{0}_from_{1}_to_{2}.csv'.format(make, from_year, to_year)
df.to_csv(file, index=False)


# In[23]:


driver.close()




