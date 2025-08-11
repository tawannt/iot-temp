import os
import pandas as pd
import requests
import json
import re
import logging

weather_data = [
    {
        "entry_id": 1,
        "Temperature": 25.5,
        "Humidity": 60,
        "created_at": "2025-08-07 08:00:00",
        "Location": "London"
    },
    {
        "entry_id": 2,
        "Temperature": 26.0,
        "Humidity": 58,
        "created_at": "2025-08-07 08:15:00",
        "Location": "Tokyo"
    },
    {
        "entry_id": 3,
        "Temperature": 24.8,
        "Humidity": 62,
        "created_at": "2025-08-07 08:30:00",
        "Location": "London"
    }
]

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face API settings
API_URL = "https://router.huggingface.co/v1/chat/completions"
try:
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
        "Content-Type": "application/json"
    }
except KeyError:
    logger.error("HF_TOKEN environment variable not set")
    raise SystemExit("Please set HF_TOKEN with your Hugging Face API key")

# CSV file settings
CSV_FILE = "data.csv"  # Replace with your CSV file path
TIMESTAMP_COL = "created_at"  # Column for timestamps (optional)
TEMP_COL = "Temperature"  # Column for temperature


def get_current_temperature(df):
    """Get the most recent temperature from the CSV."""
    if df is None or df.empty:
        return "Error: Could not load CSV data."
    if TEMP_COL not in df.columns:
        return f"Error: Column '{TEMP_COL}' not found in CSV."
    
    if TIMESTAMP_COL in df.columns:
        try:
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
            latest_entry = df.sort_values(by=TIMESTAMP_COL, ascending=False).iloc[0]
        except (ValueError, TypeError) as e:
            logger.warning(f"Timestamp parsing failed: {e}, using last row")
            latest_entry = df.iloc[-1]
    else:
        latest_entry = df.iloc[-1]
    
    temp = latest_entry[TEMP_COL]
    return f"The current temperature is {temp}°C."

def get_temperature_by_location(df, location):
    """Get temperature for a specific location."""
    if df is None or df.empty:
        return "Error: Could not load CSV data."
    if "Location" not in df.columns:
        return "Error: This CSV does not contain location data."
    
    matching_rows = df[df["Location"].str.lower() == location.lower()]
    if matching_rows.empty:
        return f"No temperature data found for {location}."
    
    if TIMESTAMP_COL in df.columns:
        try:
            matching_rows[TIMESTAMP_COL] = pd.to_datetime(matching_rows[TIMESTAMP_COL])
            latest_entry = matching_rows.sort_values(by=TIMESTAMP_COL, ascending=False).iloc[0]
        except (ValueError, TypeError):
            latest_entry = matching_rows.iloc[-1]
    else:
        latest_entry = matching_rows.iloc[-1]
    
    temp = latest_entry[TEMP_COL]
    return f"The current temperature in {location} is {temp}°C."

def query_hugging_face(user_input):
    """Query Hugging Face API to interpret user input."""
    prompt = f"""
    You are a chatbot that interprets user queries about temperature data from a CSV file.
    - If the query asks for 'current temperature' or similar (e.g., 'temperature now'), return {{"intent": "current"}}.
    - If the query specifies a location (e.g., 'temperature in London'), return {{"location": "city_name"}}.
    - For unclear or unrelated queries, return {{"intent": "unrelated"}}.

    Query: "{user_input}"
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "CohereLabs/c4ai-command-r-plus:cohere",
        "max_tokens": 50,
        "temperature": 0.5
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"API response: {result}")
        try:
            # Clean the response content by removing "Output: " prefix
            content = result["choices"][0]["message"]["content"]
            # Strip any prefix like "Output: " and ensure valid JSON
            content = re.sub(r"^(Output:\s*)?", "", content).strip()
            return json.loads(content)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse API response: {e}, content: {content}")
            return {"intent": "unrelated"}
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return {"intent": "unrelated"}

def fallback_parse_input(user_input):
    """Fallback regex-based parsing for user input."""
    user_input = user_input.lower().strip()
    if "current temperature" in user_input or "temperature now" in user_input:
        return {"intent": "current"}
    match = re.search(r"(?:temperature|weather)\s*(?:in|for|at)?\s*([\w\s]+)", user_input, re.IGNORECASE)
    if match:
        return {"location": match.group(1).strip()}
    return {"intent": "unrelated"}

def process_user_input(user_input, df):
    """Process user input using Hugging Face API with regex fallback."""
    api_response = query_hugging_face(user_input)
    logger.info(f"API parsed response: {api_response}")
    
    if api_response.get("intent") == "unrelated":
        logger.info("Falling back to regex parsing")
        api_response = fallback_parse_input(user_input)
    
    if api_response.get("intent") == "current":
        return get_current_temperature(df)
    elif "location" in api_response:
        return get_temperature_by_location(df, api_response["location"])
    else:
        return "I can provide the current temperature or temperature for a location. Try saying 'current temperature' or 'temperature in [city]'."

def main():
    # Load CSV data
    df = load_csv_data()
    if df is None:
        print(f"Error: Could not load '{CSV_FILE}'. Please check the file path and format.")
        return

    print("CSV API Chatbot (Type 'exit' to quit)")
    print("Ask for the temperature, e.g., 'current temperature' or 'temperature in London'")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        response = process_user_input(user_input, df)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()