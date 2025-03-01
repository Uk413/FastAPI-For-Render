import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, List, Any
import pytz
from dateutil import parser
import pymysql
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

def suggest_event_names(user_input: str, llm=None) -> list:
    """Generate event name suggestions based on user input."""
    if llm is None:
        llm = get_llm()
    
    uncertainty_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        Analyze the following user input: '{user_input}'.
        Determine if the input indicates uncertainty about naming an event.
        Uncertainty may include phrases like 'I don't know', 'not sure', 'can't think', or similar expressions.
        Return 'True' if the input suggests uncertainty, otherwise return 'False'.
        """),
        ("human", "Please provide the inferred response.")
    ])

    uncertainty_response = (uncertainty_prompt | llm).invoke({})
    is_uncertain = uncertainty_response.content.strip().lower() == "true"
    
    if is_uncertain:
        suggestion_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            The user is trying to name their event but is unsure. 
            Provide 3-5 random creative and relevant random tech related event name suggestions to the user.
            Just suggest some names and nothing else.
            """),
            ("human", "Please provide the event name suggestions.")
        ])
    else:
        suggestion_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            The user has provided the following event name: '{user_input}'.
            Generate 3-5 similar or refined variations of this name that are creative and professional.
            Just provide the names, one per line.
            """),
            ("human", "Please provide the event name suggestions.")
        ])
    
    response = (suggestion_prompt | llm).invoke({})
    suggestions = response.content.strip().split("\n")
    suggestions = [re.sub(r'[^\w\s]', '', suggestion).strip() for suggestion in suggestions if suggestion.strip()]
    
    return suggestions

def analyze_name_choice(message: str, original_name: str, suggestions: list, llm=None) -> str:
    """
    Use LLM to analyze user's choice between original name and suggestions.
    Returns: 'original', '1', '2', '3', '4', or 'unclear'
    """
    if llm is None:
        llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        Analyze the user's response to choose between their original name and suggestions.
        
        Original name: {original_name}
        Available suggestions:
        {', '.join(f'{i+1}. {name}' for i, name in enumerate(suggestions))}
        
        User response: {message}
        
        Task: Determine which name the user wants to use.
        
        Rules:
        1. If they mention a number (1-4) or say "first", "second", etc -> Return that number
        2. If they want their original name -> Return "original"
        3. If they say "use [name]" or "choose [name]" -> Match with suggestions and return corresponding number
        4. If unclear -> Return "unclear"
        
        Return ONLY one of these values:
        - 'original' if they want to keep their original name
        - '1', '2', '3', or '4' if they chose that numbered suggestion
        - 'unclear' if their choice is ambiguous
        
        Examples:
        "keep original" -> "original"
        "first one" -> "1"
        "use Tech Summit 2024" -> "1" (if that's suggestion #1)
        "i like number 2" -> "2"
        "choose Innovation Expo" -> "2" (if that's suggestion #2)
        "use my original name" -> "original"
        "maybe" -> "unclear"
        """),
        ("human", "What is the user's choice?")
    ])
    
    try:
        clean_message = message.strip().lower()
        
        if clean_message.isdigit():
            num = int(clean_message)
            if 1 <= num <= len(suggestions):
                return str(num)
        
        if clean_message.startswith(("use ", "choose ")):
            name_mentioned = clean_message.replace("use ", "").replace("choose ", "").strip()
            for i, suggestion in enumerate(suggestions, 1):
                if name_mentioned == suggestion.lower():
                    print(f"Matched suggestion #{i}: {suggestion}")
                    return str(i)
        
        response = (prompt | llm).invoke({})
        choice = response.content.strip().lower()
        
        print(f"LLM analyzed choice: '{choice}' for message: '{message}'")
        
        if choice in ['1', '2', '3', '4', 'original', 'unclear']:
            return choice
            
        return 'unclear'
        
    except Exception as e:
        print(f"Error in analyze_name_choice: {str(e)}")
        return 'unclear'

def parse_user_choice(user_response: str, suggestions: list, llm=None) -> str:
    if llm is None:
        llm = get_llm()
    
    for suggestion in suggestions:
        pattern = rf'\b{re.escape(suggestion)}\b'
        if re.search(pattern, user_response, re.IGNORECASE):
            return suggestion
    
    prompt = f"""
    The user has been given these event name suggestions:
    {chr(10).join(f"{i+1}. {name}" for i, name in enumerate(suggestions))}
    
    Their response is: '{user_response}'
    
    Analyze if they want to:
    1. Keep their original name (indicating phrases like "keep original", "use my name", "original", "my name", "keep it", "no change", etc.)
    2. Choose one of the suggestions (they might say things like "use suggestion X", "I like X", "X sounds good", "use the X one", etc.)
    
    If they want a suggestion, identify which one they want by number (1-{len(suggestions)}).
    
    Return EXACTLY one of these:
    - "original" if they want to keep their original name
    - The number (1-{len(suggestions)}) of the suggestion they want
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", "Please provide the inferred choice.")
    ])
    
    response = (prompt_template | llm).invoke({})
    inferred_choice = response.content.strip().lower()
    
    if inferred_choice == "original":
        return "original"
    
    try:
        index = int(inferred_choice)
        if 1 <= index <= len(suggestions):
            return suggestions[index - 1]
    except ValueError:
        numbers = re.findall(r'\d+', user_response)
        if numbers:
            try:
                index = int(numbers[0])
                if 1 <= index <= len(suggestions):
                    return suggestions[index - 1]
            except ValueError:
                pass
    
    final_prompt = f"""
    The user response '{user_response}' needs to be matched to one of these suggestions:
    {chr(10).join(f"{i+1}. {name}" for i, name in enumerate(suggestions))}
    
    Return ONLY the number (1-{len(suggestions)}) of the most likely matching suggestion.
    If unclear, return 1.
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", final_prompt),
        ("human", "Please provide the suggestion number.")
    ])
    
    final_response = (prompt_template | llm).invoke({})
    try:
        index = int(final_response.content.strip())
        if 1 <= index <= len(suggestions):
            return suggestions[index - 1]
    except ValueError:
        pass
    
    return "original"

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
    
    print(f"\n=== Starting partner info extraction for URL: {url} ===")
    
    if llm is None:
        llm = get_llm()
        
    try:
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        print(f"Normalized URL: {url}")
            
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        print(f"URL fetch successful. Status code: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.title.string.strip() if soup.title else ""
        meta_desc = soup.find('meta', {'name': 'description'}) or soup.find('meta', {'property': 'og:description'})
        description = meta_desc.get('content', '').strip() if meta_desc else ""
        
        logo = soup.find('link', {'rel': 'icon'}) or soup.find('link', {'rel': 'shortcut icon'})
        logo_url = logo.get('href', '') if logo else ''
        if logo_url and not logo_url.startswith(('http://', 'https://')):
            logo_url = f"{url.rstrip('/')}/{logo_url.lstrip('/')}"
        
        about_section = soup.find(lambda tag: tag.name in ['div', 'section'] and 
                                any(keyword in tag.get('class', []) or keyword in tag.get('id', '') 
                                    for keyword in ['about', 'company', 'who-we-are', 'profile']))
        
        if about_section:
            text_content = ' '.join([text.strip() for text in about_section.stripped_strings])[:2000]
        else:
            paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 30]
            text_content = ' '.join(paragraphs)[:2000]
            
            if len(text_content) < 200:
                text_content = ' '.join([
                    text for text in soup.stripped_strings
                    if len(text.strip()) > 20
                ])[:2000]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract accurate company information from the provided webpage content.
                Analyze the content carefully to identify:
                1. The full legal company name
                2. A shorter display name or common name
                3. When the company was established (year only)
                4. Main industry or business sector
                5. A concise description of what the company does

                Return ONLY the following JSON structure without any additional text:
                {{
                    "active": true,
                    "partnerName": "FULL_LEGAL_NAME",
                    "customUrl": "SHORT_URL_FRIENDLY_NAME",
                    "partnerLogoPath": "LOGO_URL_FROM_INPUT",
                    "partnerDisplayName": "DISPLAY_NAME",
                    "partnerEstablishmentDate": "YYYY",
                    "partnerDescription": "BRIEF_DESCRIPTION_OF_BUSINESS",
                    "partnerIndustry": "PRIMARY_INDUSTRY_CATEGORY",
                    "partnerSocialLinks": "{{\\"websiteurl\\":\\"WEBSITE_URL\\"}}"
                }}

                If certain information is not found, make a reasonable guess but DO NOT include phrases like 
                "not specified" or "not found" in the output. For establishment date, use the current year if unknown.
                """),
            ("human", f"URL: {url}\nTitle: {title}\nDescription: {description}\nLogoURL: {logo_url}\nContent: {text_content}")
        ])
        
        response = (prompt | llm).invoke({})
        json_str = response.content.strip()
        json_str = re.sub(r'^```json\s*|\s*```$', '', json_str, flags=re.MULTILINE)
        partner_info = json.loads(json_str)
        
        existing_partner = get_existing_partner(partner_info["partnerName"])
        
        if existing_partner:
            print(f"Partner already exists: {partner_info['partnerName']}")
            return existing_partner
        
        if not partner_info.get("customUrl") or partner_info["customUrl"] == "SHORT_URL_FRIENDLY_NAME":
            partner_info["customUrl"] = partner_info["partnerName"].lower().replace(" ", "-")
        
        if logo_url and partner_info.get("partnerLogoPath") == "LOGO_URL_FROM_INPUT":
            partner_info["partnerLogoPath"] = logo_url
        else:
            partner_info["partnerLogoPath"] = "NA"
        
        if isinstance(partner_info.get("partnerSocialLinks"), dict):
            partner_info["partnerSocialLinks"] = json.dumps(partner_info["partnerSocialLinks"])
        elif partner_info.get("partnerSocialLinks") == "{\"websiteurl\":\"WEBSITE_URL\"}":
            partner_info["partnerSocialLinks"] = json.dumps({"websiteurl": url})
        
        return {
            "active": True,
            "customUrl": partner_info["customUrl"][:50],
            "partnerDisplayName": partner_info["partnerDisplayName"][:50],
            "partnerEstablishmentDate": partner_info["partnerEstablishmentDate"][:4],  # Ensure only year
            "partnerLogoPath": partner_info["partnerLogoPath"],
            "partnerName": partner_info["partnerName"][:100],
            "partnerType": "INDUSTRY",
            "partnerDescription": partner_info["partnerDescription"][:200],
            "partnerIndustry": partner_info["partnerIndustry"][:50],
            "partnerSocialLinks": partner_info["partnerSocialLinks"]
        }
        
    except Exception as e:
        print(f"\n!!! Error in extract_partner_info: {str(e)}")
        domain = url.split('/')[2] if len(url.split('/')) > 2 else url
        return {
            "active": False,
            "customUrl": domain.replace(".", "-"),
            "partnerDisplayName": domain.split('.')[0].title(),
            "partnerEstablishmentDate": str(datetime.now().year),
            "partnerLogoPath": "NA",
            "partnerName": domain,
            "partnerType": "INDUSTRY",
            "partnerDescription": "Organization details could not be extracted",
            "partnerIndustry": "Technology",
            "partnerSocialLinks": json.dumps({"websiteurl": url})
        }

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
    try:
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        response = requests.head(url, allow_redirects=True, verify=False)
        return response.status_code == 200
    except:
        return False

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

def handle_name_recognition(user_input: str, llm) -> Tuple[bool, Optional[str]]:
      
    # Direct extraction attempt using regex for common patterns
    name_patterns = [
        r"(?:name it|call it|titled|named)\s+['\"](.+?)['\"]",  # "name it 'Tech Summit'"
        r"(?:name|call|title):\s+['\"]?(.+?)['\"]?$",           # "name: Tech Summit"
        r"(?:name|call|title) is\s+['\"]?(.+?)['\"]?",          # "name is Tech Summit"
        r"['\"](.+?)['\"] (?:as the name|as the title)",        # "'Tech Summit' as the name"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return True, match.group(1).strip()
    
    # Use LLM for more complex cases
    prompt = [
        SystemMessage(content="""
        Your task is to determine if the user has provided a specific name for their event or if they need help with naming.
        If they've provided a name, extract it.
        If they need help or are unsure, indicate this.
        
        Output format (JSON):
        {
            "has_provided_name": true/false,
            "extracted_name": "The name" or null
        }
        """),
        HumanMessage(content=f"User message: '{user_input}'")
    ]
    
    response = llm(prompt)
    
    try:
        # Try to extract JSON from the response
        import json
        import re
        
        # Look for JSON pattern in the response
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return result.get("has_provided_name", False), result.get("extracted_name")
    except:
        # Fallback to simple heuristic if JSON parsing fails
        uncertainty_phrases = [
            "not sure", "don't know", "help me", "suggest", "recommendation", 
            "can't decide", "undecided", "need ideas", "help with name", "thinking"
        ]
        
        if any(phrase in user_input.lower() for phrase in uncertainty_phrases):
            return False, None
            
        # If we've reached here, attempt one more extraction using the LLM
        follow_up_prompt = [
            SystemMessage(content="Extract the event name from the user's message, or respond with 'NO_NAME_PROVIDED' if none is found."),
            HumanMessage(content=user_input)
        ]
        
        name_extraction = llm(follow_up_prompt).content.strip()
        if name_extraction != "NO_NAME_PROVIDED":
            return True, name_extraction
    
    # Default fallback
    return False, None

import json
from typing import Tuple, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from src.utils import get_llm


import json
from typing import Tuple, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from src.utils import get_llm



def recognize_name_intent(user_input: str, llm) -> Tuple[bool, Optional[str]]:
    
    prompt = f"""
    User input: "{user_input}"
    
    Your task is to determine if the user has provided an event name or if they are asking for suggestions.
    
    Examples:
    - "I want to name it TechFest" → They provided the name "TechFest"
    - "Let's call it Innovate 2023" → They provided the name "Innovate 2023"
    - "I don't know what to name it" → They are uncertain, need suggestions
    - "Can you suggest some names?" → They are uncertain, need suggestions
    
    Rules:
    - If the user has CLEARLY mentioned a name, for example: "technohack","techdeva","technovate", extract it exactly as they said.
    - If there's any sign of uncertainty (e.g., "I don't know", "not sure", "help me", "can you suggest"), mark as NO_NAME_PROVIDED.
    - If the user seems to be asking a question rather than stating a name, mark as NO_NAME_PROVIDED.
    
    Return **only a valid JSON** in this format:
    {{
        "has_name": true/false,
        "extracted_name": "Event Name" or "NO_NAME_PROVIDED"
    }}
    """

    # ✅ Step 2: Make a more explicit request to the LLM
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Extract event name if given, otherwise indicate uncertainty.")])

    try:
        # ✅ Step 3: Improved error handling and response parsing
        llm_output = response.content.strip()
        
        # Try to extract JSON from potential markdown code blocks
        if "```json" in llm_output:
            # Extract JSON content from markdown code block
            json_start = llm_output.find("```json") + 7
            json_end = llm_output.find("```", json_start)
            llm_output = llm_output[json_start:json_end].strip()
        
        result = json.loads(llm_output)

        # ✅ Step 4: More explicit decision logic
        if result.get("has_name", False) and result.get("extracted_name") != "NO_NAME_PROVIDED":
            print(f"✅ Name detected: {result['extracted_name']}")
            return True, result["extracted_name"]
        
        # ✅ Step 5: Log decision for debugging
        print("❌ No name detected, user is uncertain")
        return False, None  

    except (json.JSONDecodeError, KeyError) as e:
        # ✅ Step 6: Improved error logging
        print(f"Error parsing LLM response: {e}")
        print(f"LLM output was: {llm_output}")
        return False, None



import re

from src.utils import get_llm

llm = get_llm()
session_store = {}

import json

async def handle_name_suggestion(session_id: str, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Handles event name suggestion using LLM dynamically."""
    
    if session_id not in session_store:
        session_store[session_id] = {
            "name_conversation": {"stage": "initial", "collected_info": {}, "attempts": 0},
            "suggested_names": []
        }

    session_data = session_store[session_id]
    name_conv = session_data.get("name_conversation", {"stage": "initial", "collected_info": {}, "attempts": 0})

    has_name, extracted_name = recognize_name_intent(user_input, llm)

    # If a name is directly specified in the input
    if has_name:
        context["hackathon_details"]["drillName"] = extracted_name
        
        return {
            "message": f"Great! Your event name is set to '{extracted_name}'. When do you want to organize it?",
            "context": context,
            "current_question": "phaseStartDt"  # Moving to next step
        }

    # Check if we're in the selection phase and analyze the response
    if name_conv.get("stage") == "awaiting_choice" and "suggested_names" in session_data:
        # Use the analyze_name_choice function to determine which name the user selected
        choice, selected_name = analyze_name_choice(user_input, session_data["suggested_names"], llm)
        
        if selected_name:
            # User selected a name
            context["hackathon_details"]["drillName"] = selected_name
            return {
                "message": f"Great! Your event name is set to '{selected_name}'. When do you want to organize it?",
                "context": context,
                "current_question": "phaseStartDt"  # Moving to next step
            }
        elif "new suggestion" in user_input.lower() or "different" in user_input.lower() or "other" in user_input.lower():
            # User wants new suggestions
            name_conv["stage"] = "suggesting"
            return await handle_name_suggestion(session_id, "Please suggest new names", context)
        else:
            # User response is unclear, ask for clarification
            return {
                "message": "I'm not sure which name you selected. Please either type the name you prefer or the number (e.g., '1' for the first suggestion).",
                "suggestions": session_data["suggested_names"],
                "requires_selection": True,
                "context": context,
                "current_question": "drillName"
            }

    if name_conv["stage"] == "initial":
        name_conv["stage"] = "questioning"  # Only go into questioning if no name is found

    if name_conv["stage"] == "questioning":
        if name_conv["attempts"] >= 2:  # If 2 attempts are done, move to name suggestion
            name_conv["stage"] = "suggesting"
        else:
            question_prompt = f"""
            The user is unsure about their event's name. Generate a single clarifying question
            to help them describe the event better.

            Given context so far:
            {name_conv.get("collected_info", {})}

            Rules:
            - Make the question conversational.
            - Ask about event purpose, audience, or theme.
            - Return **only the question**, nothing else.
            """

            response = llm.invoke([SystemMessage(content=question_prompt), HumanMessage(content="What question should I ask?")])
            question = response.content.strip()

            name_conv["stage"] = "collecting"
            return {
                "message": question,
                "context": context,
                "current_question": "drillName"
            }

    # Collect User Response & Ask More Questions if Needed
    if name_conv["stage"] == "collecting":
        if not name_conv.get("collected_info"):
            name_conv["collected_info"] = {}
            
        name_conv["collected_info"][f"q{name_conv.get('attempts', 0) + 1}"] = user_input
        name_conv["attempts"] = name_conv.get("attempts", 0) + 1

        if name_conv.get("attempts", 0) >= 3:  # If 3 responses collected, move to "suggesting"
            name_conv["stage"] = "suggesting"
        else:
            name_conv["stage"] = "questioning"
            return await handle_name_suggestion(session_id, user_input, context)

    # Generate Event Name Suggestions
    if name_conv["stage"] == "suggesting":
        summary_prompt = f"""
        Based on the user's responses, generate 5 creative and relevant event name suggestions.

        Event details:
        Category: {context.get("hackathon_details", {}).get("drillCategory", "")}
        Subcategory: {context.get("hackathon_details", {}).get("drillSubCategory", "")}
        User input: {name_conv.get("collected_info", {})}

        Rules:
        - Names should be professional and catchy.
        - Provide **only the names**, one per line.
        - Names should be relevant to the event type and purpose.
        - Each name should be unique and memorable.
        """

        response = llm.invoke([SystemMessage(content=summary_prompt), HumanMessage(content="Suggest event names.")])
        suggestions = [s.strip() for s in response.content.strip().split("\n") if s.strip()]
        
        # Ensure we have exactly 5 suggestions
        while len(suggestions) > 5:
            suggestions.pop()
        
        # If we somehow got less than 5, add generic ones
        while len(suggestions) < 5:
            suggestions.append(f"Innovative {context.get('hackathon_details', {}).get('drillSubCategory', 'Event')} {len(suggestions) + 1}")

        name_conv["stage"] = "awaiting_choice"
        session_data["suggested_names"] = suggestions
        session_data["name_conversation"] = name_conv

        return {
            "message": "Here are some event name suggestions:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions)),
            "suggestions": suggestions,
            "requires_selection": True,
            "context": context,
            "current_question": "drillName"
        }
    
    # Fallback response if none of the above conditions are met
    return {
        "message": "Let's give your event a name. What would you like to call it?",
        "context": context,
        "current_question": "drillName"
    }

def analyze_name_choice(message: str, suggested_names: List[str], llm) -> Tuple[int, str]:
    
    if not suggested_names:
        return None, None
        
    # Check for direct number mentions (1-5)
    for i in range(1, len(suggested_names) + 1):
        number_patterns = [
            f"^{i}$",  # Exact match (e.g., "1")
            f"^{i}[.)]",  # Number with punctuation (e.g., "1." or "1)")
            f"option {i}",  # Option with number (e.g., "option 1")
            f"number {i}",  # Number with word (e.g., "number 1")
            f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th"  # Ordinal (e.g., "1st")
        ]
        
        for pattern in number_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return i-1, suggested_names[i-1]
    
    # Check for exact name matches
    for i, name in enumerate(suggested_names):
        if name.lower() in message.lower():
            return i, name
    
    # Use LLM for more advanced analysis if no simple match found
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
            
        # Parse "NUMBER:NAME" format
        match = re.match(r"(\d+):(.*)", result)
        if match:
            index = int(match.group(1)) - 1
            name = match.group(2).strip()
            
            # Verify the name exists in our suggestions (as a sanity check)
            for i, suggested in enumerate(suggested_names):
                if suggested.lower() == name.lower() or (index == i and name in suggested):
                    return i, suggested_names[i]
        
        # If format parsing failed, check if the response contains any of the suggested names
        for i, name in enumerate(suggested_names):
            if name.lower() in result.lower():
                return i, name
    
    except Exception as e:
        print(f"Error in analyze_name_choice: {str(e)}")
    
    # Default: could not determine a selection
    return None, None


def analyze_name_choice(message: str, suggested_names: List[str], llm) -> Tuple[int, str]:
    if not suggested_names:
        return None, None
        
    # Check for direct number mentions (1-5)
    for i in range(1, len(suggested_names) + 1):
        number_patterns = [
            f"^{i}$",  # Exact match (e.g., "1")
            f"^{i}[.)]",  # Number with punctuation (e.g., "1." or "1)")
            f"option {i}",  # Option with number (e.g., "option 1")
            f"number {i}",  # Number with word (e.g., "number 1")
            f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th"  # Ordinal (e.g., "1st")
        ]
        
        for pattern in number_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return i-1, suggested_names[i-1]
    
    # Check for exact name matches
    for i, name in enumerate(suggested_names):
        if name.lower() in message.lower():
            return i, name
    
    # Use LLM for more advanced analysis if no simple match found
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
            
        # Parse "NUMBER:NAME" format
        match = re.match(r"(\d+):(.*)", result)
        if match:
            index = int(match.group(1)) - 1
            name = match.group(2).strip()
            
            # Verify the name exists in our suggestions (as a sanity check)
            for i, suggested in enumerate(suggested_names):
                if suggested.lower() == name.lower() or (index == i and name in suggested):
                    return i, suggested_names[i]
        
        # If format parsing failed, check if the response contains any of the suggested names
        for i, name in enumerate(suggested_names):
            if name.lower() in result.lower():
                return i, name
    
    except Exception as e:
        print(f"Error in analyze_name_choice: {str(e)}")
    
    # Default: could not determine a selection
    return None, None

def handle_name_selection(message: str, suggested_names: List[str], llm) -> str:
    clean_message = message.lower().strip()
    
    for name in suggested_names:
        if name.lower() in clean_message:
            return name
    
    number_map = {str(i+1): name for i, name in enumerate(suggested_names)}
    for num, name in number_map.items():
        if num in clean_message:
            return name
    
    prompt = f"""
    Determine which event name the user is selecting from the list of suggestions.
    
    User message: "{message}"
    
    Available suggestions:
    {', '.join([f"{i+1}. {name}" for i, name in enumerate(suggested_names)])}
    
    Rules:
    - If the user mentions a number (like "first one", "number 3", etc.), select that numbered option
    - If the user mentions part of a name, select the full name that contains it
    - If the user expresses preference ("I like", "I prefer", etc.) for a specific option, select it
    
    Return ONLY the full text of the selected name, exactly as written in the suggestions.
    If you cannot determine a clear selection, return "UNCLEAR".
    """
    
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Which suggestion did the user select?")
    ])
    
    selection = response.content.strip()
    
    if selection in suggested_names:
        return selection
    
    if "UNCLEAR" in selection:
        return None
    
    for name in suggested_names:
        if name.lower() in selection.lower():
            return name
    
    return None


def extract_naming_context(user_input: str, event_type: str, llm) -> Dict[str, Any]:

    prompt = f"""
    Extract information from the user's message that would be helpful for naming a {event_type}.
    Focus on topic, audience, purpose, goals, industry, and any specific keywords mentioned.
    Also determine if we have sufficient information to suggest good event names.
    
    User message: "{user_input}"
    
    Output format (JSON):
    {{
        "topic": "extracted topic or null",
        "audience": "extracted audience or null",
        "purpose": "extracted purpose or null", 
        "industry": "extracted industry or null",
        "keywords": ["keyword1", "keyword2", ...],
        "tone_preference": "formal/creative/technical/etc or null",
        "sufficient_info": true/false
    }}
    """
    
    try:
        response = llm.invoke([{"role": "system", "content": prompt}])
        result = extract_json_from_response(response.content)
        if result:
            return result
    except Exception as e:
        print(f"Error extracting context with LLM: {str(e)}")
    
    # Default fallback
    return {
        "topic": None,
        "audience": None,
        "purpose": None,
        "industry": None,
        "keywords": [],
        "tone_preference": None,
        "sufficient_info": False
    }

def ask_contextual_question(name_conv: Dict[str, Any], context: Dict[str, Any], event_type: str, llm) -> str:

    extracted_info = name_conv["extracted_info"]
    
    prompt = f"""
    I'm helping a user name a {event_type}. Based on the information I have so far, generate a single focused question
    to gather more relevant naming context. Make the question conversational and natural.
    
    Current information:
    - Event type: {event_type}
    - Topic: {extracted_info.get('topic')}
    - Audience: {extracted_info.get('audience')}
    - Purpose: {extracted_info.get('purpose')}
    - Industry: {extracted_info.get('industry')}
    - Keywords: {', '.join(extracted_info.get('keywords', []))}
    
    What's missing is information about: {', '.join([k for k, v in extracted_info.items() if not v and k != 'sufficient_info' and k != 'keywords'])}
    
    Generate only ONE question to gather the most important missing information. Keep it concise:
    """
    
    try:
        response = llm.invoke([{"role": "system", "content": prompt}])
        question = response.content.strip()
        # Remove any leading assistant formatting that might be included
        question = re.sub(r'^(Assistant: |Question: )', '', question)
        return question
    except Exception as e:
        print(f"Error generating question with LLM: {str(e)}")
    
    # Default fallback questions based on the event stage
    if name_conv["attempts"] == 0:
        return f"To help name your {event_type}, could you tell me more about the main focus or topic?"
    else:
        return "What's the target audience or key goal for this event?"

def suggest_names_based_on_context(name_conv: Dict[str, Any], context: Dict[str, Any], event_type: str, llm) -> str:

    extracted_info = name_conv["extracted_info"]
    
    # Prepare a detailed prompt with all available context
    info_parts = []
    for key, value in extracted_info.items():
        if value and key != "sufficient_info" and key != "keywords":
            info_parts.append(f"- {key.capitalize()}: {value}")
    
    if extracted_info.get("keywords"):
        info_parts.append(f"- Keywords: {', '.join(extracted_info['keywords'])}")
    
    context_str = "\n".join(info_parts)
    
    prompt = f"""
    Generate 2-3 catchy, memorable names for a {event_type} based on the following information:
    
    {context_str}
    
    The names should be concise, relevant, and clearly convey the purpose of the event.
    Each suggestion should be different in style or approach.
    
    Output format (JSON):
    {{
        "suggestions": [
            {{
                "name": "Suggestion 1",
                "rationale": "Brief explanation of why this name works"
            }},
            {{
                "name": "Suggestion 2",
                "rationale": "Brief explanation of why this name works"
            }},
            ...
        ]
    }}
    """
    
    try:
        response = llm.invoke([{"role": "system", "content": prompt}])
        result = extract_json_from_response(response.content)
        
        if result and "suggestions" in result and len(result["suggestions"]) > 0:
            suggestions = result["suggestions"]
            name_conv["suggested_names"] = suggestions
            name_conv["stage"] = "suggestion"
            
            # Format the response with the suggestions
            response_parts = ["Based on what you've told me, here are some name suggestions for your event:"]
            
            for i, suggestion in enumerate(suggestions):
                response_parts.append(f"{i+1}. \"{suggestion['name']}\" - {suggestion['rationale']}")
            
            response_parts.append("\nDo any of these names work for you? Or would you like me to generate more options?")
            return "\n\n".join(response_parts)
    except Exception as e:
        print(f"Error generating suggestions with LLM: {str(e)}")
    
    # Fallback suggestion if the LLM call fails
    default_suggestion = f"Innovation {event_type}"
    name_conv["suggested_names"] = [{"name": default_suggestion, "rationale": "A simple, clear name for your event."}]
    name_conv["stage"] = "suggestion"
    
    return f"I suggest naming your event: \"{default_suggestion}\". Does this work for you? (Or would you like more suggestions?)"

def evaluate_name_feedback(user_input: str, name_conv: Dict[str, Any], context: Dict[str, Any], llm) -> str:

    suggested_names = [s["name"] for s in name_conv["suggested_names"]]
    
    for name in suggested_names:
        # Check if user explicitly mentioned a name (case insensitive, allowing for some variation)
        if name.lower() in user_input.lower() or re.search(rf"\b{re.escape(name.lower())}\b", user_input.lower()):
            name_conv["final_name"] = name
            name_conv["stage"] = "completed"
            context["hackathon_details"]["drillName"] = name
            return f"Great! I've set the event name to '{name}'."
    
    # Use LLM to interpret complex feedback
    prompt = f"""
    Analyze the user's response to these event name suggestions:
    {', '.join('"' + s["name"] + '"' for s in name_conv["suggested_names"])}
    
    User response: "{user_input}"
    
    Output format (JSON):
    {{
        "liked_any": true/false,
        "selected_name": "specific name they liked or null",
        "wants_more_suggestions": true/false,
        "provided_own_name": true/false,
        "own_name": "their provided name or null",
        "sentiment": "positive/neutral/negative"
    }}
    """
    
    try:
        response = llm.invoke([{"role": "system", "content": prompt}])
        result = extract_json_from_response(response.content)
        
        if result:
            # If user liked one of the suggestions
            if result.get("liked_any") and result.get("selected_name"):
                name_conv["final_name"] = result["selected_name"]
                name_conv["stage"] = "completed"
                context["hackathon_details"]["drillName"] = result["selected_name"]
                return f"Great! I've set the event name to '{result['selected_name']}'."
                
            # If user provided their own name
            elif result.get("provided_own_name") and result.get("own_name"):
                name_conv["final_name"] = result["own_name"]
                name_conv["stage"] = "completed"
                context["hackathon_details"]["drillName"] = result["own_name"]
                return f"Great! I've set the event name to '{result['own_name']}'."
                
            # If user wants more suggestions
            elif result.get("wants_more_suggestions"):
                # Generate new suggestions with explicit instructions to make them different
                event_type = context["hackathon_details"].get("drillSubCategory", "Event")
                
                prompt = f"""
                Generate 2-3 NEW and DIFFERENT catchy, memorable names for a {event_type}.
                Make them notably different from previous suggestions: {', '.join('"' + s["name"] + '"' for s in name_conv["suggested_names"])}
                
                Context information:
                {', '.join([f"{k}: {v}" for k, v in name_conv["extracted_info"].items() if v and k != "sufficient_info"])}
                
                Output format (JSON):
                {{
                    "suggestions": [
                        {{
                            "name": "New Suggestion 1",
                            "rationale": "Brief explanation of why this name works"
                        }},
                        ...
                    ]
                }}
                """
                
                try:
                    new_response = llm.invoke([{"role": "system", "content": prompt}])
                    new_result = extract_json_from_response(new_response.content)
                    
                    if new_result and "suggestions" in new_result and len(new_result["suggestions"]) > 0:
                        name_conv["suggested_names"] = new_result["suggestions"]
                        
                        response_parts = ["Here are some more name options for your event:"]
                        for i, suggestion in enumerate(new_result["suggestions"]):
                            response_parts.append(f"{i+1}. \"{suggestion['name']}\" - {suggestion['rationale']}")
                        
                        response_parts.append("\nDo any of these names work better for you?")
                        return "\n\n".join(response_parts)
                except Exception as e:
                    print(f"Error generating new suggestions: {str(e)}")
    except Exception as e:
        print(f"Error evaluating feedback: {str(e)}")
    
    # Fallback: If we couldn't clearly interpret the response, ask more directly
    return "Would you like to go with one of the suggested names, or would you prefer to provide your own name for the event?"

def extract_json_from_response(response_text: str) -> Dict[str, Any]:

    try:
        # Try to find a JSON block in the response
        json_match = re.search(r'```json\n(.*?)\n```|```(.*?)```|\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1) or json_match.group(2) or json_match.group(0)
            # Clean up the JSON string
            json_str = re.sub(r'```json|```', '', json_str)
            return json.loads(json_str)
        
        # If no JSON block found, try parsing the entire response
        return json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from: {response_text}")
        return None

def clean_json_response(response: str) -> str:
    
    response = re.sub(r'```json\s*|```\s*', '', response)
    
    response = response.strip()
    
    if not response.startswith('{'):
        json_start = response.find('{')
        if json_start != -1:
            response = response[json_start:]
    
    if not response.endswith('}'):
        json_end = response.rfind('}')
        if json_end != -1:
            response = response[:json_end + 1]
    
    return response


def generate_names_by_subcategory(subcategory: str, llm=None) -> List[str]:

    if llm is None:
        llm = get_llm()
    
    prompt = f"""
    Based on the subcategory '{subcategory}', suggest 5 event names that would be suitable for this category.
    """
    
    response = llm.invoke(prompt)
    subcategory_names = response.content.strip().split('\n')
    
    return [name.strip() for name in subcategory_names if name.strip()]

def generate_similar_names(name: str, llm=None) -> List[str]:
    if llm is None:
        llm = get_llm()
    
    prompt = f"""
    Suggest similar names for the event name '{name}'. Provide a list of 5 names that are similar in style or theme.
    """
    
    response = llm.invoke(prompt)
    similar_names = response.content.strip().split('\n')
    
    return [name.strip() for name in similar_names if name.strip()]

def handle_user_name_choice(user_input: str, suggestions: List[str], original_name: str, llm=None) -> str:
    if llm is None:
        llm = get_llm()
    

    if user_input.lower() in ["keep original", "original", "my name"]:
        return original_name
    

    prompt = f"""
    The user has provided the following input: '{user_input}'.
    Determine if the user wants to keep their original name '{original_name}' or choose from the suggestions: {suggestions}.
    Respond with 'keep_original' if they want to keep the original name, or return the chosen suggestion if they select one.
    If the input is unclear, respond with 'ambiguous'.
    """
    
    response = llm.invoke(prompt)
    inferred_intent = response.content.strip()
    
    if inferred_intent == "keep_original":
        return original_name
    elif inferred_intent in suggestions:
        return inferred_intent
    else:
        return None