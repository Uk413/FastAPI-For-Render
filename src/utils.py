import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from playwright.async_api import async_playwright
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, List, Any
import pytz
from dateutil import parser
import pymysql
import asyncio
from src.constants import CATEGORY_SUBCATEGORY_MAP
from bs4 import BeautifulSoup
import requests
import json
load_dotenv()

def get_llm():
    """Create and return a ChatGoogleGenerativeAI instance with proper authentication."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=google_api_key  
    )

llm = get_llm()

class DatabaseHandler:
    def __init__(self):
        
        self.host = os.getenv("DB_HOST")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.database = os.getenv("DB_NAME")
        self.connection = None

    def connect(self):
        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
        except pymysql.MySQLError as e:
            print(f"Error connecting to MySQL: {e}")

    def close(self):
        
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def create_table_if_not_exists(self, table_name: str, columns: list):
        if not self.connection:
            self.connect()

        column_definitions = []
        for col in columns:
            if col == "isDrillPaid":
                column_definitions.append(f"`{col}` TINYINT(1)")
            elif col == "phaseStartDt":
                column_definitions.append(f"`{col}` VARCHAR(10)")
            else:
                column_definitions.append(f"`{col}` VARCHAR(255)")

        columns_sql = ", ".join(column_definitions)
        
        query = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            {columns_sql},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                self.connection.commit()
                print(f"Table {table_name} created or verified successfully")
        except pymysql.MySQLError as e:
            print(f"Error creating table: {e}")
            raise

    def insert_record(self, table_name: str, record: dict):

        if not self.connection:
            self.connect()

        columns = ", ".join([f"`{key}`" for key in record.keys()])
        placeholders = ", ".join(["%s"] * len(record))
        query = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders});"

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, list(record.values()))
                self.connection.commit()
                print(f"Record inserted successfully into {table_name}")
        except pymysql.MySQLError as e:
            print(f"Error inserting record: {e}")
            raise

class DateHandler:
    
    def __init__(self, default_timezone: str = "UTC"):
        self.default_timezone = pytz.timezone(default_timezone)
        self.llm = get_llm()
        
    def parse_date_string(self, date_string: str) -> Optional[datetime]:
        """Parse date string using LLM and fallback to traditional parsing."""
        try:
            
            date_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""
                Convert the following natural language date input into a specific date in DD-MM-YYYY format.
                Input: {date_string}
                
                Rules:
                - If the year is not specified, use the current year
                - If month is not specified, use the next occurrence of that date
                - For relative dates (tomorrow, next week, etc.), calculate from current date
                - Handle phrases like "5th may", "7 march", "next monday", "tomorrow", "next week"
                - If input is ambiguous, make a reasonable assumption and pick the next occurrence
                - Return ONLY the date in DD-MM-YYYY format or "INVALID" if cannot be interpreted
                
                Current date: {datetime.now().strftime('%d-%m-%Y')}
                """),
                ("human", "Please convert the date.")
            ])
            
            response = (date_prompt | self.llm).invoke({})
            llm_date = response.content.strip()
            
            if llm_date != "INVALID":
                try:
                    return datetime.strptime(llm_date, "%d-%m-%Y")
                except ValueError:
                    pass
            
            
            parsed_date = parser.parse(date_string, dayfirst=True, fuzzy=True)
            
          
            if parsed_date.year == 1900:
                current_year = datetime.now().year
                parsed_date = parsed_date.replace(year=current_year)
            
            
            current_date = datetime.now()
            if parsed_date.date() < current_date.date():
                if parsed_date.month < current_date.month:
                    parsed_date = parsed_date.replace(year=current_year + 1)
                elif parsed_date.month == current_date.month and parsed_date.day < current_date.day:
                    parsed_date = parsed_date.replace(year=current_year + 1)
            
            return parsed_date
            
        except (ValueError, TypeError):
            
            formats = [
                "%d-%m-%Y", "%d/%m/%Y", "%m-%d-%Y", "%m/%d/%Y",
                "%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%m.%d.%Y",
                "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_string, fmt)
                except ValueError:
                    continue
            
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""
                The user has provided this date input: '{date_string}'
                Please interpret this as a future date, considering:
                - Current date is {datetime.now().strftime('%d-%m-%Y')}
                - If it's a relative date (tomorrow, next week), calculate accordingly
                - If it's ambiguous, choose the next occurrence
                - Return the date in DD-MM-YYYY format or "INVALID"
                """),
                ("human", "Please provide the interpreted date.")
            ])
            
            final_response = (final_prompt | self.llm).invoke({})
            final_date = final_response.content.strip()
            
            if final_date != "INVALID":
                try:
                    return datetime.strptime(final_date, "%d-%m-%Y")
                except ValueError:
                    pass
            
            return None

    def validate_date(self, date_string: str, timezone_str: str = None) -> Tuple[bool, str, Optional[str]]:
        """Validate the parsed date."""
        date_string = re.sub(r'\s+', ' ', date_string.strip())
        
        parsed_date = self.parse_date_string(date_string)
        if not parsed_date:
            return False, "", "I couldn't understand that date format. Please provide a date like '7 March' or 'next Monday'."
            
        current_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        if parsed_date.replace(tzinfo=timezone.utc) < current_date:
            return False, "", "The date cannot be in the past. Please provide a future date."
            
        max_future_date = current_date + timedelta(days=730)
        if parsed_date.replace(tzinfo=timezone.utc) > max_future_date:
            return False, "", "The date cannot be more than 2 years in the future."
            
        formatted_date = parsed_date.strftime("%d-%m-%Y")
        
        return True, formatted_date, None
        
    def get_phase_dates(self, phase_start_date: str, timezone_str: str) -> Dict[str, str]:
        
        tz = pytz.timezone(timezone_str)
        
        phase_start = datetime.strptime(phase_start_date, "%d-%m-%Y")
        phase_start = tz.localize(phase_start.replace(hour=0, minute=0, second=0, microsecond=0))
        
        phase_end = phase_start + timedelta(days=15)
        
        drill_registration_start = datetime.now(tz).replace(microsecond=0)
        
        drill_registration_end = phase_start - timedelta(days=1)
        
        return {
            "phaseStartDt": phase_start.strftime("%d-%m-%Y"),
            "phaseEndDt": phase_end.strftime("%d-%m-%Y"),
            "drillRegistrationStartDt": drill_registration_start.strftime("%d-%m-%Y"),
            "drillRegistrationEndDt": drill_registration_end.strftime("%d-%m-%Y")
        }
        
    def get_timezone_offset(self, timezone_str: str) -> str:
        
        tz = pytz.timezone(timezone_str)
        offset = tz.utcoffset(datetime.now())
        hours = int(offset.total_seconds() // 3600)
        minutes = int((offset.total_seconds() % 3600) // 60)
        return f"{'+' if hours >= 0 else '-'}{abs(hours):02d}:{minutes:02d}"

def auto_correct_input(field_name: str, user_input: str, llm=None) -> str:
    """Use LLM to auto-correct typos in user input."""
    if llm is None:
        llm = get_llm()
    correction_prompt = f"""
    The user entered '{user_input}' for the field '{field_name}'. 
    If there are any typos or errors in the input, correct it and provide the most likely intended value.
    If the input is already correct, return it as-is.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", correction_prompt),
        ("human", "Please provide the corrected value.")
    ])
    response = (prompt_template | llm).invoke({})
    return response.content.strip()

def generate_drill_description(drill_info: dict, llm=None) -> str:
    """Generate a short description for the event based on the drillPurpose."""
    if llm is None:
        llm = get_llm()

    drill_purpose = drill_info.get("drillPurpose", "Innovation")  

    description_prompt = f"""
    Generate a short and engaging description for the following event:
    - Name: {drill_info["drillName"]}
    - Type: {drill_info["drillType"]}
    - Purpose: {drill_purpose}
    
    The description should clearly reflect the purpose of the event and provide an overview that aligns with the event's goals.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", description_prompt),
        ("human", "Please generate the description based on the above details.")
    ])

    response = (prompt_template | llm).invoke({})
    return response.content.strip()

def infer_purpose(state: dict, user_response: str, llm) -> str:
    """Use LLM to infer the purpose of the event."""
    subcategory = state["hackathon_details"].get("drillSubCategory", "").upper()
    prompt = f"""
    Based on the drill subcategory '{subcategory}' and the user input: '{user_response}',
    determine the most likely purpose of the event.
    The possible purposes are 'Innovation' or 'Hiring'.
    If the input strongly suggests one of these purposes, return it.
    Otherwise, default to 'Innovation'.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", "Provide the inferred purpose.")
    ])
    response = (prompt_template | llm).invoke({})
    inferred_purpose = response.content.strip().capitalize()
    return inferred_purpose if inferred_purpose in ["Innovation", "Hiring"] else "Innovation"

def infer_subcategory(user_response: str, category_subcategory_map: dict, llm=None) -> str:
    if llm is None:
        llm = get_llm()
    """Use LLM to infer the most relevant subcategory."""
    valid_subcategories = list(category_subcategory_map.keys())
    prompt = f"""
    Given the user input: '{user_response}', determine the most relevant subcategory from the following list:
    {valid_subcategories}.
    If no clear match is found, return an empty string.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", "Provide the inferred subcategory.")
    ])
    response = (prompt_template | llm).invoke({})
    inferred_subcategory = response.content.strip().upper()
    return inferred_subcategory if inferred_subcategory in category_subcategory_map else ""

def infer_yes_no(user_response: str, llm=None) -> str:
    """Use LLM to infer whether the user's response is 'Yes' or 'No'."""
    if llm is None:
        llm = get_llm()
    prompt = f"""
    Given the user input: '{user_response}', determine whether the response indicates 'Yes' or 'No'.
    Return 'Yes' if the input strongly suggests affirmation, otherwise return 'No'.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", "Provide the inferred response.")
    ])
    response = (prompt_template | llm).invoke({})
    inferred_response = response.content.strip().capitalize()
    return inferred_response if inferred_response in ["Yes", "No"] else "No"

def check_for_cancellation(user_response: str, llm=None) -> bool:
    if llm is None:
        llm = get_llm()
    """Use LLM to infer if the user wants to cancel the registration process."""
    prompt = f"""
    Given the user input: '{user_response}', determine whether the user wants to cancel the registration process.
    Return 'True' if the input strongly suggests cancellation (e.g., 'cancel', 'stop', 'don't want to proceed'), otherwise return 'False'.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", "Provide the inferred response.")
    ])
    response = (prompt_template | llm).invoke({})
    inferred_response = response.content.strip().capitalize()
    return inferred_response == "True"

def infer_user_uncertainty(user_input: str, llm=None) -> bool:
    if llm is None:
        llm = get_llm()
    
    prompt = """
    Analyze the following user input: '{input}'
    
    Determine if the user is expressing uncertainty about naming their event.
    Look for:
    - Direct expressions of uncertainty ("I'm not sure", "I don't know")
    - Indirect uncertainty ("help me name", "can you suggest")
    - Questions about naming ("what should I call it?")
    - Requests for suggestions or assistance
    
    Respond with only 'True' if uncertainty is detected, 'False' otherwise.
    """
    
    response = llm.invoke(prompt.format(input=user_input))
    return response.content.strip().lower() == "true"

def generate_random_event_names(llm=None) -> list:
    """Generate random creative event names when the user is unsure."""
    if llm is None:
        llm = get_llm()
    
    prompt = """
    Generate 5 random, creative, and professional event names.
    Only return the names, each on a new line.
    """
    
    response = llm.invoke(prompt)
    suggestions = response.content.strip().split("\n")
    return [name.strip() for name in suggestions if name.strip()]

def generate_drill_overview_description(drill_info: dict, llm=None) -> str:
    """Generate an overview description for the event."""
    if llm is None:
        llm = get_llm()

    description_prompt = f"""
    Generate a single, concise overview paragraph (max 2-3 sentences) for this event:
    Name: {drill_info["drillName"]}
    Type: {drill_info["drillType"]}
    Purpose: {drill_info["drillPurpose"]}
    Start Date: {drill_info["phaseStartDt"]}
    End Date: {drill_info["phaseEndDt"]}
    Registration Start Date: {drill_info["drillRegistrationStartDt"]}
    Registration End Date: {drill_info["drillRegistrationEndDt"]}
    Is Paid: {"Yes" if drill_info["isDrillPaid"] else "No"}

    Rules:
    - Create only ONE paragraph
    - Focus on the event's purpose and key details
    - Keep it professional and engaging
    - Do not list dates or repeat information verbatim
    - Do not use bullet points or multiple paragraphs
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", description_prompt),
        ("human", "Generate a single overview paragraph.")
    ])
    response = (prompt_template | llm).invoke({})
    return response.content.strip()

def generate_faq_answers(drill_info: dict, llm=None) -> list:
    """Generate FAQ answers based on the drill details."""
    drill_title = drill_info.get("drillName", "the hackathon")
    is_paid = drill_info.get("isDrillPaid", False)
    
    def format_date(date_str):
        if not date_str:
            return ""
        try:
            if "T" in date_str: 
                dt = datetime.strptime(date_str.split('T')[0], "%Y-%m-%d")
                return dt.strftime("%d %B %Y")
            else:  
                dt = datetime.strptime(date_str, "%d-%m-%Y")
                return dt.strftime("%d %B %Y")
        except Exception:
            return date_str

    drill_phase = json.loads(drill_info.get("drillPhase", "{}")) if isinstance(drill_info.get("drillPhase"), str) else drill_info.get("drillPhase", {})
    phase_schedule = drill_phase.get("schedule", [{}])[0] if drill_phase.get("schedule") else {}
    
    phase_start_dt = format_date(phase_schedule.get("phaseStartDt") or drill_info.get("phaseStartDt", ""))
    phase_end_dt = format_date(phase_schedule.get("phaseEndDt") or drill_info.get("phaseEndDt", ""))
    registration_start_dt = format_date(drill_info.get("drillRegistrationStartDt", ""))
    registration_end_dt = format_date(drill_info.get("drillRegistrationEndDt", ""))

    payment_type = "paid" if is_paid else "free"
    payment_question = "Is this a free hackathon?"
    payment_answer = (
        "<p>No, this is a paid hackathon.</p>" if is_paid else 
        "<p>Yes, this is a free hackathon. There is no registration fee required to participate.</p>"
    )

    if phase_start_dt and phase_end_dt:
        start_date_answer = (
            f"The hackathon will start on {phase_start_dt} and end on {phase_end_dt}. "
            f"Registration is open from {registration_start_dt} to {registration_end_dt}."
        )
    else:
        start_date_answer = "Start date to be announced"

    print(f"Debug - Dates: Start={phase_start_dt}, End={phase_end_dt}, Reg Start={registration_start_dt}, Reg End={registration_end_dt}")
    print(f"Debug - Final answer: {start_date_answer}")

    faq_list = [
        {
            "answer": "<p>To Participate – Click on the \"Login & Participate\" button and sign up using your email or login with your existing account. If you are already logged in then click on the \"Participate\" button to continue to join the hackathon.</p>",
            "drillId": drill_info.get("drillId", ""),
            "positionOrder": 0,
            "question": f"How do I participate in {drill_title}?",
            "questionId": ""
        },
        {
            "answer": payment_answer,
            "drillId": drill_info.get("drillId", ""),
            "positionOrder": 1,
            "question": payment_question,
            "questionId": ""
        },
        {
            "answer": "<p>Yes</p>",
            "drillId": drill_info.get("drillId", ""),
            "positionOrder": 2,
            "question": "Will I receive a certificate for participation?",
            "questionId": ""
        },
        {
            "answer": f"<p>{start_date_answer}</p>",
            "drillId": drill_info.get("drillId", ""),
            "positionOrder": 3,
            "question": "When will the hackathon start?",
            "questionId": ""
        }
    ]

    return faq_list

def extract_partner_info(url: str, llm=None) -> dict:
    """
    Extracts partner information from a given URL with improved error handling.
    Includes Playwright-based scraping and LLM processing.
    """
    print(f"\n=== Starting partner info extraction for URL: {url} ===")
    
    if llm is None:
        llm = get_llm()
        
    try:
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        print(f"Normalized URL: {url}")

        # Try Playwright scraping first
        try:
            # Run async Playwright in sync context
            html_content = asyncio.run(async_scrape_with_playwright(url))
                
            # Process with LLM
            llm_response = process_with_llm(html_content, url)
            
            # Try to parse the LLM response
            try:
                partner_info = json.loads(llm_response)
                print("Successfully extracted info using Playwright and LLM")
                return partner_info
            except json.JSONDecodeError:
                print("Failed to parse LLM response, falling back to BeautifulSoup")
                
        except Exception as e:
            print(f"Playwright scraping failed: {str(e)}, falling back to requests")
            
        # Fallback to original requests-based scraping
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        })
        
        # Fetch page content
        response = session.get(url, timeout=15, verify=False)
        response.raise_for_status()
        print(f"URL fetch successful. Status code: {response.status_code}")
        
        # Parse content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract basic info
        title = soup.title.string.strip() if soup.title else ""
        domain = url.split('/')[2] if len(url.split('/')) > 2 else url
        domain = domain.replace("www.", "")
        
        # Extract metadata
        meta_desc = (
            soup.find('meta', {'name': 'description'}) or 
            soup.find('meta', {'property': 'og:description'})
        )
        description = meta_desc.get('content', '').strip() if meta_desc else ""
        
        # Extract logo
        logo = (
            soup.find('link', {'rel': 'icon'}) or 
            soup.find('link', {'rel': 'shortcut icon'})
        )
        logo_url = logo.get('href', '') if logo else ''
        if logo_url and not logo_url.startswith(('http://', 'https://')):
            logo_url = f"{url.rstrip('/')}/{logo_url.lstrip('/')}"
        
        # Fallback info if extraction fails
        default_info = {
            "active": True,
            "customUrl": domain.replace(".", "-"),
            "partnerDisplayName": domain.split('.')[0].title(),
            "partnerEstablishmentDate": str(datetime.now().year),
            "partnerLogoPath": logo_url or "NA",
            "partnerName": title or domain.split('.')[0].title(),
            "partnerType": "INDUSTRY",
            "partnerDescription": description or "Technology company",
            "partnerIndustry": "Technology",
            "partnerSocialLinks": json.dumps({"websiteurl": url})
        }
        
        # If page content is available, try to extract more detailed info
        if response.text:
            about_section = soup.find(lambda tag: tag.name in ['div', 'section'] and 
                                    any(keyword in (tag.get('class', []) + [tag.get('id', '')])
                                        for keyword in ['about', 'company', 'who-we-are', 'profile']))
            
            if about_section:
                text_content = ' '.join([text.strip() for text in about_section.stripped_strings])[:2000]
            else:
                paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 30]
                text_content = ' '.join(paragraphs)[:2000]
            
            if text_content:
                try:
                    partner_info = default_info.copy()
                    partner_info.update({
                        "partnerDescription": text_content[:200],
                    })
                    return partner_info
                except Exception as e:
                    print(f"Error updating partner info: {str(e)}")
                    
        return default_info
        
    except Exception as e:
        print(f"Error in extract_partner_info: {str(e)}")
        # Return basic info based on URL
        domain = url.split('/')[2] if len(url.split('/')) > 2 else url
        domain = domain.replace("www.", "")
        return {
            "active": True,
            "customUrl": domain.replace(".", "-"),
            "partnerDisplayName": domain.split('.')[0].title(),
            "partnerEstablishmentDate": str(datetime.now().year),
            "partnerLogoPath": "NA",
            "partnerName": domain.split('.')[0].title(),
            "partnerType": "INDUSTRY",
            "partnerDescription": "Technology company",
            "partnerIndustry": "Technology",
            "partnerSocialLinks": json.dumps({"websiteurl": url})
        }
    
async def async_scrape_with_playwright(url: str) -> str:
    """
    Asynchronously scrape content using Playwright.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=60000)
        html_content = await page.content()
        await browser.close()
        return html_content

def process_with_llm(html_content: str, company_url: str) -> str:
    """
    Process HTML content with LLM to extract structured company information.
    """
    prompt = f"""
    Analyze the following HTML page source and extract structured company details in JSON format:
    - Company Name
    - Establishment Year
    - Industry Type
    - Description
    - Logo URL (if available)
    
    Given HTML Content: ```{html_content[:2000]}```
    
    Return JSON format as follows:
    {{
      "active": true,
      "customUrl": "<shortened-company-name>",
      "partnerDisplayName": "<company name>",
      "partnerEstablishmentDate": "<year>",
      "partnerLogoPath": "<logo-url or NA>",
      "partnerName": "<full legal name>",
      "partnerType": "<type>",
      "partnerDescription": "<description>",
      "partnerIndustry": "<industry>",
      "partnerSocialLinks": {{
        "websiteurl": "{company_url}"
      }}
    }}
    """
    
    messages = [{"role": "user", "content": prompt}]
    response = llm.invoke(messages)
    return response.content

def get_existing_partner(partner_name: str) -> Optional[dict]:
    
    api_url = os.getenv("PARTNER_URL")
    
    try:
        normalized_name = partner_name.lower().strip()
        
        response = requests.get(f"{api_url}?name={partner_name}")
        response.raise_for_status()
        partners = response.json()
        
        if partners and isinstance(partners, list):
            for partner in partners:
                if partner.get("partnerName", "").lower().strip() == normalized_name:
                    print(f"Found exact partner match for: {partner_name}")
                    return partner
            
            for partner in partners:
                if normalized_name in partner.get("partnerName", "").lower() or \
                   partner.get("partnerName", "").lower() in normalized_name:
                    print(f"Found fuzzy partner match: {partner.get('partnerName')} for query: {partner_name}")
                    return partner
        
        domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', partner_name)
        if domain_match:
            domain = domain_match.group(1)
            response = requests.get(f"{api_url}")
            response.raise_for_status()
            all_partners = response.json()
            
            if all_partners and isinstance(all_partners, list):
                for partner in all_partners:
                    partner_links = partner.get("partnerSocialLinks", "{}")
                    if isinstance(partner_links, str):
                        try:
                            links_dict = json.loads(partner_links)
                            website = links_dict.get("websiteurl", "")
                            if domain in website:
                                print(f"Found partner match by domain: {domain}")
                                return partner
                        except json.JSONDecodeError:
                            pass
        
        print(f"No existing partner found for: {partner_name}")
        return None
    except Exception as e:
        print(f"Error checking existing partner: {str(e)}")
        return None

def register_partner(partner_info: dict) -> dict:

    print(f"\n=== Starting partner registration ===")
    
    api_url = os.getenv("PARTNER_URL")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        response = requests.post(api_url, json=partner_info, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"\n!!! Error in register_partner: {str(e)}")
        raise

def validate_partner_url(url: str) -> bool:
    """
    Validates if a given URL is accessible and returns a valid response.
    """
    try:
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        
        # Configure session with custom headers and settings
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Make request with extended timeout and ignore SSL verification
        response = session.get(
            url, 
            timeout=15,
            verify=False,
            allow_redirects=True
        )
        
        # Check if response is successful and contains HTML content
        return (response.status_code == 200 and 
                'text/html' in response.headers.get('Content-Type', '').lower())
    
    except Exception as e:
        print(f"URL validation error: {str(e)}")
        return False

def find_partner_by_url(url: str) -> Optional[dict]:
    """
    Find a partner by URL in the social links
    """
    api_url = os.getenv("PARTNER_URL")
    
    try:
        # Normalize the URL for comparison
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        
        # Remove trailing slash if present
        url = url.rstrip('/')
        
        # Get all partners
        response = requests.get(api_url)
        response.raise_for_status()
        partners = response.json()
        
        if not partners or not isinstance(partners, list):
            return None
        
        # First try exact URL match
        for partner in partners:
            social_links = partner.get("partnerSocialLinks", "{}")
            if isinstance(social_links, str):
                try:
                    links_dict = json.loads(social_links)
                    website = links_dict.get("websiteurl", "").rstrip('/')
                    
                    # Try different variations of the URL
                    url_variations = [
                        url,
                        url.replace("https://", "http://"),
                        url.replace("http://", "https://"),
                        url.replace("www.", ""),
                        url + "/"
                    ]
                    
                    if website and any(website == variation for variation in url_variations):
                        print(f"Found exact partner match by URL: {website}")
                        return partner
                except json.JSONDecodeError:
                    pass
        
        # Try domain match if exact match fails
        domain = url.split('/')[2] if len(url.split('/')) > 2 else url
        domain = domain.replace("www.", "")
        
        for partner in partners:
            social_links = partner.get("partnerSocialLinks", "{}")
            if isinstance(social_links, str):
                try:
                    links_dict = json.loads(social_links)
                    website = links_dict.get("websiteurl", "")
                    if domain in website or domain.split('.')[0] in website:
                        print(f"Found partner match by domain: {domain}")
                        return partner
                except json.JSONDecodeError:
                    pass
        
        print(f"No partner found for URL: {url}")
        return None
    except Exception as e:
        print(f"Error finding partner by URL: {str(e)}")
        return None

def infer_eligibility(user_response: str, llm=None) -> list:
    
    if llm is None:
        from src.utils import get_llm
        llm = get_llm()
    
    valid_categories = ["Graduate", "Working Professionals", "College Students"]
    
    user_response_lower = user_response.lower().strip()
    
    if any(term in user_response_lower for term in ["all", "everyone", "anybody", "anyone"]):
        return valid_categories
    
    matched_categories = []
    
    if "graduate" in user_response_lower or "graduates" in user_response_lower:
        matched_categories.append("Graduate")
        
    if "college" in user_response_lower or "students" in user_response_lower:
        matched_categories.append("College Students")
        
    if "working" in user_response_lower or "professional" in user_response_lower or "professionals" in user_response_lower:
        matched_categories.append("Working Professionals")
    
    if matched_categories:
        return matched_categories
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the user's response and determine which categories of participants are eligible for an event.
        
        Valid categories:
        - Graduate
        - College Students
        - Working Professionals
        
        Instructions:
        1. Return ONLY the valid categories that match the user's response
        2. Be specific - only include categories that are clearly mentioned or implied
        3. If the user mentions "students" without further specification, interpret as "College Students"
        4. If the user mentions "professionals" or "working people", interpret as "Working Professionals"
        5. If the user mentions "graduates", interpret as "Graduate"
        6. If the intent is unclear, return the most likely category based on the response
        7. Return the result as a JSON array with exact category names as shown above
        
        Examples:
        - "Students and professionals" → ["College Students", "Working Professionals"]
        - "Only graduates" → ["Graduate"]
        - "Anyone with a degree" → ["Graduate", "Working Professionals"]
        - "Students in college" → ["College Students"]
        """), 
        ("human", user_response)
    ])
    
    try:
        response = (prompt | llm).invoke({})
        
        if response is None or not hasattr(response, 'content') or not response.content:
            print("LLM returned null or empty response")
            return ["College Students"]
            
        content = response.content.strip()
        
        if not content:
            print("LLM returned empty content")
            return ["College Students"]
        
        categories = []
        try:
            if "[" in content and "]" in content:
                start = content.find("[")
                end = content.rfind("]") + 1
                json_str = content[start:end]
                parsed_content = json.loads(json_str)
                if parsed_content is None:
                    return ["College Students"]
                categories = parsed_content
            else:
                parsed_content = json.loads(content)
                if parsed_content is None:
                    return ["College Students"]
                categories = parsed_content
        except json.JSONDecodeError:
            cleaned = content.replace("The eligible categories are:", "")
            cleaned = cleaned.replace("Based on the response, the eligible categories are:", "")
            if "," in cleaned:
                categories = [cat.strip() for cat in cleaned.split(",")]
            elif "\n" in cleaned:
                categories = [cat.strip() for cat in cleaned.split("\n") if cat.strip()]
            else:
                categories = [cleaned.strip()]
        
        if not categories:
            print("Failed to extract categories from LLM response")
            return ["College Students"]
        
        validated_categories = []
        for cat in categories:
            if cat is None or not isinstance(cat, str) or not cat.strip():
                continue
                
            matches = [valid for valid in valid_categories if valid.lower() == cat.lower()]
            if matches:
                validated_categories.append(matches[0])
            elif "college" in cat.lower() or "student" in cat.lower():
                if "College Students" not in validated_categories:
                    validated_categories.append("College Students")
            elif "work" in cat.lower() or "professional" in cat.lower():
                if "Working Professionals" not in validated_categories:
                    validated_categories.append("Working Professionals")
            elif "grad" in cat.lower():
                if "Graduate" not in validated_categories:
                    validated_categories.append("Graduate")
        
        return validated_categories if validated_categories else ["College Students"]

    except Exception as e:
        print(f"Error in infer_eligibility: {str(e)}")
        return ["College Students"]


def recognize_name_intent(user_input: str, llm) -> Tuple[bool, Optional[str]]:
    prompt = f"""
    User input: "{user_input}"
    
    Your task is to determine if the user has provided an event name or if they are asking for suggestions.
    
    Examples:
    - "I want to name it TechFest" â†’ They provided the name "TechFest"
    - "Let's call it Innovate 2023" â†’ They provided the name "Innovate 2023"
    - "I don't know what to name it" â†’ They are uncertain, need suggestions
    - "Can you suggest some names?" â†’ They are uncertain, need suggestions
    
    Rules:
    - If the user has CLEARLY mentioned a name, for example: "technohack","techdeva","technovate", extract  it exactly as they said.
    - If there's any sign of uncertainty (e.g., "I don't know", "not sure", "help me", "can you suggest"), mark as NO_NAME_PROVIDED.
    - If the user seems to be asking a question rather than stating a name, mark as NO_NAME_PROVIDED.
    
    Return **only a valid JSON** in this format:
    {{
        "has_name": true/false,
        "extracted_name": "Event Name" or "NO_NAME_PROVIDED"
    }}
    Ensure that you capture the entire name, including any additional parts after a colon (e.g., "DataBex : Data Science And AI" should be captured as "DataBex : Data Science And AI").
    """

    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Extract event name if given, otherwise indicate uncertainty.")])

    try:
        llm_output = response.content.strip()
        if "```json" in llm_output:
            json_start = llm_output.find("```json") + 7
            json_end = llm_output.find("```", json_start)
            llm_output = llm_output[json_start:json_end].strip()
        
        result = json.loads(llm_output)

        if result.get("has_name", False) and result.get("extracted_name") != "NO_NAME_PROVIDED":
            print(f"Name detected: {result['extracted_name']}")
            return True, result["extracted_name"]
        
        print("No name detected, user is uncertain")
        return False, None  

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing LLM response: {e}")
        print(f"LLM output was: {llm_output}")
        return False, None

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False, None
    
session_store = {}

def is_requesting_more_suggestions(user_input: str, llm) -> bool:
    
    prompt = f"""
    Analyze the following user input and determine if the user is asking for more options or alternatives.
    
    User Input: "{user_input}"
    
    Rules:
    - Respond with ONLY "True" if the user is asking for more suggestions.
    - Respond with ONLY "False" otherwise.
    - Interpret phrases like "more", "new suggestion", "different", "other options", "don't like", "something else" as requests for more suggestions.
    - Be lenient in interpretation but avoid false positives.
    """
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Is the user asking for more suggestions?")])
    return response.content.strip().lower() == "true"


async def handle_name_suggestion(session_id: str, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
    if session_id not in session_store:
        session_store[session_id] = {
            "name_conversation": {
                "stage": "initial",
                "is_name_finalized": False,
                "llm_active": True,
                "previous_suggestions": [],
                "attempts": 0
            }
        }

    session_data = session_store[session_id]
    name_conv = session_data["name_conversation"]

    if context["hackathon_details"].get("drillName"):
        return {
            "message": f"Your event name is set to '{context['hackathon_details']['drillName']}'. When do you want to organize it?",
            "context": context,
            "current_question": "phaseStartDt"
        }

    if name_conv["previous_suggestions"]:
        prompt = f"""
        The user was given these event name suggestions:
        {', '.join(name_conv["previous_suggestions"])}

        Now, the user responded with: "{user_input}"

        Determine if they are expressing preference for one of the names. Examples:
        - "I like Next Gen AI" â†’ selects "Next Gen AI"
        - "Let's go with AI Thrive Workshop" â†’ selects "AI Thrive Workshop"
        - "I prefer AI Launchpad" â†’ selects "AI Launchpad"
        - "Give me more suggestions" â†’ NO selection, request new names
        - "None of these work" â†’ NO selection, request new names

        Return ONLY the name they selected (if any), otherwise return "NO_SELECTION".
        """
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Which name did the user choose?")])
        selected_name = response.content.strip()

        if selected_name != "NO_SELECTION":
            context["hackathon_details"]["drillName"] = selected_name
            name_conv["is_name_finalized"] = True
            name_conv["llm_active"] = False  # Stop LLM after name selection

            return {
                "message": f"Great! {selected_name} is your event name. When do you want to organize it?",
                "context": context,
                "current_question": "phaseStartDt"
            }

    if is_requesting_more_suggestions(user_input, llm):
        name_conv["attempts"] += 1
        if name_conv["attempts"] > 2:
            return {
                "message": "I've already given multiple suggestions. Let me know which one you like!",
                "context": context,
                "current_question": "drillName"
            }

        prompt = f"""
        The user wants more event name suggestions. Generate a fresh set of names that are **different** from:
        {name_conv["previous_suggestions"]}.
        
        Keep them **short, catchy, and relevant to** "{context.get('hackathon_details', {}).get('drillSubCategory', 'event')}".  
        """
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Generate different event name suggestions.")])
        llm_output = response.content.strip()

        name_conv["previous_suggestions"] = llm_output.split("\n")
        return {
            "message": f"Here are some new ideas:\n\n{llm_output}\n\nDo any of these work for you?",
            "context": context,
            "current_question": "drillName"
        }

    name_conv["llm_active"] = True

    prompt = f"""
    The user needs help naming their event. Generate **5-7 catchy, relevant event names** based on the event details.

    **User input so far:** "{user_input}"

    **Steps:**  
    1. Extract **3-5 important keywords** from the user's description.  
    2. Generate **5-7 short, catchy event names** using these keywords.  
    3. Provide the names in a simple list format.  
    4. End with the question: **"Do any of these work for you?"**  
    """

    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Generate event name suggestions.")])
    llm_output = response.content.strip()

    name_conv["previous_suggestions"] = llm_output.split("\n")

    return {
        "message": llm_output,
        "context": context,
        "current_question": "drillName"
    }

def analyze_name_choice(message: str, suggested_names: List[str], llm) -> Tuple[int, str]:
    if not suggested_names:
        return None, None

    message_lower = message.lower()

    for i in range(1, len(suggested_names) + 1):
        number_patterns = [
            f"^{i}$",
            f"^{i}[.)]",
            f"option {i}",
            f"number {i}",
            f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th"
        ]
        for pattern in number_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return i - 1, suggested_names[i - 1]

    ordinal_mappings = {
        "first": 0,
        "second": 1,
        "third": 2,
        "fourth": 3,
        "fifth": 4,
        "last": len(suggested_names) - 1
    }
    for word, index in ordinal_mappings.items():
        if word in message_lower:
            return index, suggested_names[index]

    for i, name in enumerate(suggested_names):
        if name.lower() in message_lower:
            return i, name

    preference_keywords = ["i like", "i prefer", "let's go with", "choose", "select"]
    for keyword in preference_keywords:
        if keyword in message_lower:
            for i, name in enumerate(suggested_names):
                if name.lower() in message_lower:
                    return i, name

    prompt = f"""
    Determine which event name the user is selecting from the list of suggestions.
    
    User message: "{message}"
    
    Available suggestions:
    {', '.join([f"{i+1}. {name}" for i, name in enumerate(suggested_names)])}
    
    Rules:
    - If the user mentions a number or position (like "first one", "number 3", "the last one", etc.), select that numbered option
    - If the user mentions a name or part of a name, select the full name that contains it
    - If the user expresses preference ("I like", "I prefer", "let's go with", etc.) for a specific option, select it
    - Look for partial name matches, even if the user only mentions distinctive words from the name
    
    Respond with ONLY the number and name in the format "NUMBER:NAME", or "NONE" if no selection can be determined.
    For example: "1:Tech Summit 2025" or "NONE"
    """
    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Which suggestion did the user select?")
        ])
        result = response.content.strip()

        if "NONE" in result:
            return None, None

        match = re.match(r"(\d+):(.*)", result)
        if match:
            index = int(match.group(1)) - 1
            name = match.group(2).strip()
            for i, suggested in enumerate(suggested_names):
                if suggested.lower() == name.lower() or (index == i and name in suggested):
                    return i, suggested_names[i]

        for i, name in enumerate(suggested_names):
            if name.lower() in result.lower():
                return i, name

    except Exception as e:
        print(f"Error in analyze_name_choice: {str(e)}")

    return None, None
