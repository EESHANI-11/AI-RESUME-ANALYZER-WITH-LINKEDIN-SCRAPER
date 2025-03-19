import time
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cohere
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException

import warnings
warnings.filterwarnings('ignore')

# Streamlit Config
def streamlit_config():
    st.set_page_config(page_title='Resume Analyzer AI', layout="wide")
    st.markdown("""
        <style>
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
        </style>
    """, unsafe_allow_html=True)
    st.markdown(f'<h1 style="text-align: center;">Resume Analyzer AI</h1>', unsafe_allow_html=True)

# Resume Analyzer Class
class ResumeAnalyzer:

    @staticmethod
    def pdf_to_chunks(pdf):
        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() for page in pdf_reader.pages)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, length_function=len)
        return text_splitter.split_text(text=text)

    @staticmethod
    def analyze_with_cohere(cohere_api_key, analyze):
        co = cohere.Client(cohere_api_key)
        response = co.chat(
            model="command-r-plus",
            message=analyze,
            temperature=0.7
        )
        return response.text

    @staticmethod
    def generate_prompt(type, query_with_chunks):
        prompts = {
            "summary": f"""Provide a detailed summary of the following resume:
            {query_with_chunks}""",
            "strength": f"""Analyze and explain the strengths of this resume:
            {query_with_chunks}""",
            "weakness": f"""Analyze the weaknesses of this resume and suggest improvements:
            {query_with_chunks}"""
        }
        return prompts[type]

    @staticmethod
    def analyze_resume(type):
        with st.form(key=type.capitalize()):
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            cohere_api_key = st.text_input(label='Enter Cohere API Key', type='password')
            submit = st.form_submit_button(label='Submit')

        if submit:
            if pdf and cohere_api_key:
                try:
                    with st.spinner('Processing...'):
                        pdf_chunks = ResumeAnalyzer.pdf_to_chunks(pdf)
                        summary = ResumeAnalyzer.analyze_with_cohere(
                            cohere_api_key=cohere_api_key,
                            analyze=ResumeAnalyzer.generate_prompt("summary", " ".join(pdf_chunks))
                        )

                        if type in ["strength", "weakness"]:
                            result = ResumeAnalyzer.analyze_with_cohere(
                                cohere_api_key=cohere_api_key,
                                analyze=ResumeAnalyzer.generate_prompt(type, summary)
                            )
                        else:
                            result = summary

                    st.markdown(f'<h4 style="color: orange;">{type.capitalize()}:</h4>', unsafe_allow_html=True)
                    st.write(result)

                except Exception as e:
                    st.markdown(f'<h5 style="text-align: center;color: orange;">Error: {e}</h5>', unsafe_allow_html=True)
            else:
                st.warning("Please upload your resume and enter an API key.")

# LinkedIn Job Scraper
class LinkedInScraper:

    @staticmethod
    def webdriver_setup():
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)
        driver.maximize_window()
        return driver

    @staticmethod
    def get_user_input():
        with st.form(key='linkedin_scraper'):
            st.write("🔍 Search Jobs on LinkedIn")
            job_title = st.text_input("Job Title (comma-separated)", "")
            job_location = st.text_input("Job Location", "India")
            job_count = st.number_input("Number of Jobs", min_value=1, value=5, step=1)
            submit = st.form_submit_button(label="Search")
        
        return job_title.split(','), job_location, job_count, submit

    @staticmethod
    def build_url(job_title, job_location):
        job_title_query = '%2C%20'.join([title.strip().replace(" ", "%20") for title in job_title])
        return f"https://www.linkedin.com/jobs/search?keywords={job_title_query}&location={job_location}"

    @staticmethod
    def open_link(driver, link):
        while True:
            try:
                driver.get(link)
                driver.implicitly_wait(5)
                time.sleep(3)
                driver.find_element(By.CSS_SELECTOR, 'span.switcher-tabs__placeholder-text.m-auto')
                return
            except NoSuchElementException:
                continue

    @staticmethod
    def scrape_jobs(driver, job_count):
        jobs = []
        job_cards = driver.find_elements(By.CSS_SELECTOR, '.base-card__full-link')

        for job in job_cards[:job_count]:
            jobs.append({
                "Title": job.text,
                "Link": job.get_attribute("href")
            })
        
        return jobs

    @staticmethod
    def main():
        driver = None
        try:
            job_title, job_location, job_count, submit = LinkedInScraper.get_user_input()
            
            if submit:
                if job_title and job_location:
                    with st.spinner('🔍 Searching Jobs on LinkedIn...'):
                        driver = LinkedInScraper.webdriver_setup()
                        search_url = LinkedInScraper.build_url(job_title, job_location)
                        LinkedInScraper.open_link(driver, search_url)
                        job_listings = LinkedInScraper.scrape_jobs(driver, job_count)
                    
                    if job_listings:
                        st.write("### 📌 Job Listings")
                        for job in job_listings:
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #ddd; padding: 10px; border-radius: 10px; margin-bottom: 10px; background-color: #f9f9f9;">
                                    <h4 style="color: #004182; margin-bottom: 5px;">
                                        <a href="{job['Link']}" target="_blank" style="text-decoration: none; color: #004182;">🔹 {job['Title']}</a>
                                    </h4>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                    else:
                        st.warning("⚠ No jobs found. Try different keywords.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

        finally:
            if driver:
                driver.quit()

# Streamlit UI
streamlit_config()

with st.sidebar:
    option = option_menu('', ['Summary', 'Strength', 'Weakness', 'LinkedIn Jobs'],
                         icons=['file-text', 'award', 'exclamation-triangle', 'linkedin'])

if option == 'Summary':
    ResumeAnalyzer.analyze_resume("summary")
elif option == 'Strength':
    ResumeAnalyzer.analyze_resume("strength")
elif option == 'Weakness':
    ResumeAnalyzer.analyze_resume("weakness")
elif option == 'LinkedIn Jobs':
    LinkedInScraper.main()
