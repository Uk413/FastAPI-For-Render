from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from datetime import datetime
import requests
from dotenv import load_dotenv
from src.utils import (
    DatabaseHandler,
    DateHandler,
    get_llm,
    infer_subcategory,
    infer_yes_no,
    generate_drill_description,
    suggest_event_names,
    parse_user_choice,
    infer_purpose,
    generate_faq_answers,
    analyze_name_choice,
    extract_partner_info,
    register_partner,
    validate_partner_url,
    handle_name_suggestion,
    infer_eligibility,
    recognize_name_intent,
    handle_name_selection
)
from src.constants import DEFAULT_DRILL_INFO, CATEGORY_SUBCATEGORY_MAP
import json
import uuid

load_dotenv()
app = FastAPI()

date_handler = DateHandler(default_timezone="Asia/Kolkata")
db_handler = DatabaseHandler()
llm = get_llm()

class SessionInput(BaseModel):
    session_id: Optional[str] = None
    message: str
    context: Optional[Dict[str, Any]] = None
    current_question: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    message: str
    suggestions: Optional[List[str]] = None
    event_link: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    current_question: Optional[str] = None
    requires_selection: bool = False
    registration_complete: bool = False

session_store: Dict[str, Dict[str, Any]] = {}

@app.post("/chat", response_model=SessionResponse)
async def chat_endpoint(chat_input: SessionInput):
    try:
        session_id = chat_input.session_id or str(uuid.uuid4())
        
        if session_id not in session_store:
            session_store[session_id] = {
                "context": {"hackathon_details": DEFAULT_DRILL_INFO.copy()},
                "current_question": None
            }
        
        session_data = session_store[session_id]
        message = chat_input.message
        context = chat_input.context or session_data["context"]
        current_question = chat_input.current_question or session_data["current_question"]

        if not current_question:
            session_data.update({
                "context": context,
                "current_question": "drillSubCategory"
            })
            return SessionResponse(
                session_id=session_id,
                message="What is the subcategory of your event? (e.g., Workshop, Webinar, Innovation Hackathon, Hiring Hackathon, etc.)",
                context=context,
                current_question="drillSubCategory"
            )

        if current_question == "drillSubCategory":
            subcategory = infer_subcategory(message, CATEGORY_SUBCATEGORY_MAP, llm)
            if not subcategory:
                return SessionResponse(
                    session_id=session_id,
                    message="Could not determine a valid subcategory. Please try again.",
                    context=context,
                    current_question="drillSubCategory"
                )
            context["hackathon_details"]["drillSubCategory"] = subcategory
            context["hackathon_details"]["drillCategory"] = CATEGORY_SUBCATEGORY_MAP[subcategory]
            
            session_data.update({
                "context": context,
                "current_question": "drillName"
            })
            return SessionResponse(
                session_id=session_id,
                message="What name do you want to give to your event?",
                context=context,
                current_question="drillName"
            )

        if current_question == "drillName":
            if "suggested_names" in session_store.get(session_id, {}) and session_store[session_id].get("name_conversation", {}).get("stage") == "awaiting_choice":
                # Use the enhanced analyze_name_choice function to determine selection
                choice_index, selected_name = analyze_name_choice(message, session_store[session_id]["suggested_names"], llm)
                
                if selected_name:
                    # User has made a clear selection
                    context["hackathon_details"]["drillName"] = selected_name
                    session_data.update({
                        "context": context,
                        "current_question": "phaseStartDt"  # Move to date question
                    })
                    session_store[session_id] = session_data  # Update session store
                    
                    return SessionResponse(
                        session_id=session_id,
                        message=f"Great! Your event name is set to '{selected_name}'. When do you want to organize it?",
                        context=context,
                        current_question="phaseStartDt"
                    )
                elif "new" in message.lower() or "different" in message.lower() or "other" in message.lower():
                    # User wants new suggestions - reset the suggestion state
                    if "name_conversation" in session_store[session_id]:
                        session_store[session_id]["name_conversation"]["stage"] = "suggesting"
                else:
                    # Selection was unclear, continue with name suggestion flow
                    pass
            
            # Check for direct name in message
            has_name, extracted_name = recognize_name_intent(message, llm)

            if has_name:
                context["hackathon_details"]["drillName"] = extracted_name
                session_data.update({
                    "context": context,
                    "current_question": "phaseStartDt"  # Move to date question
                })
                session_store[session_id] = session_data  # Update session store
                
                return SessionResponse(
                    session_id=session_id,
                    message=f"Great! Your event name is set to '{extracted_name}'. When do you want to organize it?",
                    context=context,
                    current_question="phaseStartDt"
                )

            # If we reach here, process through the name suggestion workflow
            name_response = await handle_name_suggestion(session_id, message, context)
            
            # Update session data with the response
            session_data["context"] = name_response["context"]
            session_data["current_question"] = name_response["current_question"]
            session_store[session_id] = session_data  # Update session store
            
            return SessionResponse(
                session_id=session_id,
                message=name_response["message"],
                suggestions=name_response.get("suggestions"),
                context=name_response["context"],
                current_question=name_response["current_question"],
                requires_selection=name_response.get("requires_selection", False)
            )

        elif current_question == "phaseStartDt":
            parsed_date = date_handler.parse_date_string(message)
            
            if parsed_date is None:
                return SessionResponse(
                    session_id=session_id,
                    message="I couldn't understand that date format. Please provide a date in DD-MM-YYYY format.",
                    context=context,
                    current_question="phaseStartDt"
                )
            
            formatted_date = parsed_date.strftime("%d-%m-%Y")
            
            is_valid, transformed_date, error_message = date_handler.validate_date(formatted_date)
            if not is_valid:
                return SessionResponse(
                    session_id=session_id,
                    message=error_message,
                    context=context,
                    current_question="phaseStartDt"
                )

            phase_dates = date_handler.get_phase_dates(transformed_date, "Asia/Kolkata")
            context["hackathon_details"].update(phase_dates)
            
            return SessionResponse(
                session_id=session_id,
                message="Do you want submissions based on specific themes or solutions to problems? (Problems Based or Theme Based)",
                context=context,
                current_question="drillType"
            )

        elif current_question == "drillType":
            drill_type = "Theme Based" if "theme" in message.lower() else "Product Based"
            context["hackathon_details"]["drillType"] = drill_type
            return SessionResponse(
                session_id=session_id,
                message="Will this event be paid? (Yes/No)",
                context=context,
                current_question="isDrillPaid"
            )

        elif current_question == "isDrillPaid":
            is_paid = infer_yes_no(message, llm)
            context["hackathon_details"]["isDrillPaid"] = is_paid == "Yes"
            return SessionResponse(
                session_id=session_id,
                message="What is the purpose of this event? (Provide a one-liner answer)",
                context=context,
                current_question="drillPurpose"
            )

        elif current_question == "drillPurpose":
            purpose = infer_purpose(context, message, llm)
            context["hackathon_details"]["drillPurpose"] = purpose
            
            session_data.update({
                "context": context,
                "current_question": "eligibility"
            })
            return SessionResponse(
                session_id=session_id,
                message="Who can participate in this event? (Options: Graduate, College Students, Working Professionals)",
                context=context,
                current_question="eligibility"
            )

        elif current_question == "eligibility":
            eligibility = infer_eligibility(message, llm)
            if not eligibility:
                return SessionResponse(
                    session_id=session_id,
                    message="Could not determine eligibility. Please specify who can participate (Graduate, College Students, Working Professionals)",
                    context=context,
                    current_question="eligibility"
                )
            context["hackathon_details"]["eligibility"] = eligibility
            
            session_data.update({
                "context": context,
                "current_question": "hasPartner"
            })
            return SessionResponse(
                session_id=session_id,
                message="Do you want to add a partner organization for this event? (Yes/No)",
                context=context,
                current_question="hasPartner"
            )

        elif current_question == "hasPartner":
            has_partner = infer_yes_no(message, llm)
            context["has_partner"] = has_partner == "Yes"
            
            if context["has_partner"]:
                session_data.update({
                    "context": context,
                    "current_question": "partnerUrl"
                })
                return SessionResponse(
                    session_id=session_id,
                    message="Please provide the website URL of the partner organization:",
                    context=context,
                    current_question="partnerUrl"
                )
            else:

                context["hackathon_details"].update({
                    "drillPartnerId": "b886470e-52ac-4d34-8621-ff3e4d8335fb",
                    "drillPartnerName": "WUElev8 Innovation services private ltd"
                })
                
                try:
                    drill_info = context["hackathon_details"]
                    drill_info["drillDescription"] = generate_drill_description(drill_info, llm)
                    
                    api_payload = prepare_api_payload(drill_info)
                    event_link = register_event(api_payload, drill_info["eligibility"])
                    
                    save_to_database(drill_info)
                    session_store.pop(session_id, None)
                    
                    return SessionResponse(
                        session_id=session_id,
                        message="Your event has been successfully registered!",
                        event_link=event_link,
                        context=context,
                        registration_complete=True
                    )
                except Exception as e:
                    print(f"Registration error: {str(e)}")
                    return SessionResponse(
                        session_id=session_id,
                        message=f"An error occurred during registration: {str(e)}",
                        context=context,
                        current_question=current_question
                    )

        elif current_question == "partnerUrl":
            partner_url = message.strip()
            
            try:
                # Validate URL first
                if not validate_partner_url(partner_url):
                    return SessionResponse(
                        session_id=session_id,
                        message="The provided URL seems to be invalid or inaccessible. Please provide a valid URL.",
                        context=context,
                        current_question="partnerUrl"
                    )

                # Extract partner information
                partner_info = extract_partner_info(partner_url, llm)
                
                # Check if this is a new partner or an existing one
                partner_id = None
                partner_name = None
                
                if "partnerId" in partner_info:
                    # This is an existing partner returned by get_existing_partner
                    partner_id = partner_info["partnerId"]
                    partner_name = partner_info["partnerDisplayName"] or partner_info["partnerName"]
                    print(f"Using existing partner: {partner_name} (ID: {partner_id})")
                else:
                    # Ensure social links are properly formatted
                    if isinstance(partner_info.get('partnerSocialLinks'), str):
                        social_links = partner_info['partnerSocialLinks']
                    else:
                        social_links = json.dumps({"websiteurl": partner_url})
                    partner_info['partnerSocialLinks'] = social_links
                    
                    # Register new partner
                    partner_response = register_partner(partner_info)
                    
                    if not partner_response or "partnerId" not in partner_response:
                        raise ValueError("Failed to register partner")
                    
                    partner_id = partner_response["partnerId"]
                    partner_name = partner_response["partnerDisplayName"]
                    print(f"Registered new partner: {partner_name} (ID: {partner_id})")
                
                # Update event details with partner information
                context["hackathon_details"].update({
                    "drillPartnerId": partner_id,
                    "drillPartnerName": partner_name
                })
                
                # Generate description and prepare payload
                drill_info = context["hackathon_details"]
                drill_info["drillDescription"] = generate_drill_description(drill_info, llm)
                
                api_payload = prepare_api_payload(drill_info)
                event_link = register_event(api_payload, drill_info["eligibility"])

                # Save to database and clean up session
                save_to_database(drill_info)
                session_store.pop(session_id, None)
                
                return SessionResponse(
                    session_id=session_id,
                    message=f"Event successfully registered with partner {partner_name}!",
                    event_link=event_link,
                    context=context,
                    registration_complete=True
                )
                
            except Exception as e:
                print(f"Partner registration error: {str(e)}")
                return SessionResponse(
                    session_id=session_id,
                    message="There was an issue with the partner registration. Please try again with a different URL.",
                    context=context,
                    current_question="partnerUrl"
                )

    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def prepare_api_payload(drill_info: dict) -> dict:
    
    phase_start_date = datetime.strptime(drill_info["phaseStartDt"], "%d-%m-%Y")
    phase_end_date = datetime.strptime(drill_info["phaseEndDt"], "%d-%m-%Y")
    registration_start_date = datetime.strptime(drill_info["drillRegistrationStartDt"], "%d-%m-%Y")
    registration_end_date = datetime.strptime(drill_info["drillRegistrationEndDt"], "%d-%m-%Y")

    drill_phase = {
        "type": "Single",
        "hasIdeaPhase": "",
        "isSelfPaced": False,
        "schedule": [{
            "phaseType": "HACKATHON",
            "phaseDesc": "Hackathon Phase",
            "dateConfirmed": True,
            "phaseStartDt": phase_start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "phaseEndDt": phase_end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "phaseSubmissionEndDt": None,
            "isSubmissionAllowed": False,
            "phaseName": "Phase 1",
            "phaseTimezone": "Asia/Kolkata",
            "phaseMode": "Online",
            "phasePosition": 0,
        }]
    }
    
    partner_id = drill_info.get("drillPartnerId", "b886470e-52ac-4d34-8621-ff3e4d8335fb")
    partner_name = drill_info.get("drillPartnerName", "WUElev8 Innovation services private ltd")

    return {
        "drillCategory": drill_info["drillCategory"],
        "drillSubCategory": drill_info["drillSubCategory"],
        "drillNature": "",
        "drillDescription": drill_info["drillDescription"],
        "drillName": drill_info["drillName"],
        "drillPurpose": drill_info["drillPurpose"],
        "drillTimezone": "Asia/Kolkata",
        "drillType": drill_info["drillType"],
        "drillScheduleIfNotDateKnown": "",
        "drillId": None,
        "drillInvitationType": "PUBLIC",
        "isDrillPaid": drill_info["isDrillPaid"],
        "drillPhase": json.dumps(drill_phase),
        "selectedSubCategoryLabel": drill_info["drillSubCategory"],
        "drillPartnerId": partner_id,
        "drillPartnerName": partner_name,
        "drillRegistrationStartDt": registration_start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "drillRegistrationEndDt": registration_end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    }

def register_event(payload: dict, eligibility_data) -> str:
    print(f"\n=== Starting event registration ===")
    session_id = str(uuid.uuid4())
    payload["session_id"] = session_id
    
    api_url = os.getenv("DRILL_URL")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        print(f"Sending payload to API: {json.dumps(payload, indent=2)}")
        
        response = requests.post(api_url, json=payload, headers=headers)
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        print(f"Response text: {response.text}")
        
        response.raise_for_status()
        api_response = response.json()
        
        print(f"API Response: {json.dumps(api_response, indent=2)}")
        
        cust_url = api_response.get("drillCustUrl")
        drill_id = api_response.get("drillId")

        if not cust_url or not drill_id:
            raise ValueError(f"Invalid API response: missing drillCustUrl or drillId. Response: {api_response}")
        
        eligibility_json = None
        if isinstance(eligibility_data, list):
            eligibility_json = json.dumps(eligibility_data)
        elif isinstance(eligibility_data, str):
            try:
                json.loads(eligibility_data)
                eligibility_json = eligibility_data
            except json.JSONDecodeError:
                eligibility_json = json.dumps([item.strip() for item in eligibility_data.split(',')])
        else:
            eligibility_json = json.dumps(["Graduate", "Working Professionals", "College Students"])
        
        overview_payload = {
            "drillId": drill_id,
            "eligibility": eligibility_json,   
            "guidelines": None,
            "domain": None,
            "industry": None,
            "totalPrizeValue": None,
            "createdTs": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00",
            "updatedTs": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00",
            "createdBy": None,
            "updatedBy": None,
            "overviewSectionComplete": None,
            "drillDescription": payload["drillDescription"]
        }
        
        print(f"Sending overview payload: {json.dumps(overview_payload, indent=2)}")
        
        overview_api_url = os.getenv("OVERVIEW_URL")
        overview_response = requests.post(overview_api_url, json=overview_payload, headers=headers)
        overview_response.raise_for_status()

        drill_info = payload.copy()
        drill_info["drillId"] = drill_id
        faq_answers = generate_faq_answers(drill_info, llm)

        faq_api_url = os.getenv("FAQ_URL")
        for faq in faq_answers:
            print(f"Sending FAQ payload: {json.dumps(faq, indent=2)}")
            faq_response = requests.post(faq_api_url, json=faq, headers=headers)
            faq_response.raise_for_status()
            print(f"FAQ Response status code: {faq_response.status_code}")
            print(f"FAQ Response text: {faq_response.text}")

        event_link = f"https://dev.whereuelevate.com/drills/{cust_url}"
        print(f"Generated event link: {event_link}")
        
        return event_link
    
    except requests.exceptions.RequestException as e:
        print(f"API Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"API Error Response: {e.response.text}")
            print(f"Status code: {e.response.status_code}")
            print(f"Headers: {e.response.headers}")
        raise HTTPException(
            status_code=500, 
            detail=f"API Error: {str(e)}\nResponse: {e.response.text if hasattr(e, 'response') and e.response is not None else 'No response'}"
        )
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def save_to_database(drill_info: dict):
    try:
  
        required_columns = [
            "drillSubCategory",
            "drillName",
            "phaseStartDt",
            "drillType",
            "isDrillPaid",
            "drillPurpose"
        ]

        db_handler.create_table_if_not_exists("event_records", required_columns)

        filtered_record = {}
        for column in required_columns:
            value = drill_info.get(column)
            if column == "isDrillPaid":
                filtered_record[column] = 1 if value else 0
            elif column == "phaseStartDt":
                if isinstance(value, datetime):
                    filtered_record[column] = value.strftime("%d-%m-%Y")
                else:
                    filtered_record[column] = value
            else:
                filtered_record[column] = value

        db_handler.insert_record("hackathon_records", filtered_record)
        print(f"Successfully saved record to database: {filtered_record}")

    except Exception as e:
        error_msg = f"Database Error: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get the current status of a session."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_store[session_id]

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a session from memory."""
    if session_id in session_store:
        session_store.pop(session_id)
    return {"message": "Session cleared"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
