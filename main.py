import dotenv
dotenv.load_dotenv()

import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")




import os
import csv
# Removed direct Groq import
import dotenv
# import faiss
import easyocr # Import EasyOCR again
# Removed pytesseract import
# Removed PIL.Image import
import docx
import pypdf
import argparse
# Removed json import as we'll use structured output directly
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Any # Removed Union as return type will be BaseModel or None

# Import Langchain components
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# Removed JsonOutputParser as with_structured_output handles parsing/validation
from langchain_core.runnables import Runnable  # Import Runnable

dotenv.load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Initialize Clients ---
llm = None
if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY environment variable not set.")
    # Exit gracefully if API key is missing when script is run directly
    if __name__ == "__main__":
        exit(1)
    # In a web context, llm remains None, and functions will check for it.
else:
    # Initialize ChatGroq client (Langchain)
    # Initialize it here if API key is available
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0) # Use temperature=0 for more consistent output

# EasyOCR reader will be initialized in the extract_text_from_png function


# --- Pydantic Model for Structured Output ---
class RentalAgreementInfo(BaseModel):
    """
    Extracted information from a rental agreement matching the required CSV format.
    """
    # Changed to accept int or str based on observed Groq output and desired CSV
    agreement_value: Optional[Any] = Field(None, description="The numerical value of the rent amount in integer format only, no currency symbols or commas.")
    agreement_start_date: Optional[str] = Field(None, description="The start date of the lease agreement in DD.MM.YYYY format.")
    agreement_end_date: Optional[str] = Field(None, description="The end date of the lease agreement in DD.MM.YYYY format.")
    # Changed to accept int or str
    renewal_notice_days: Optional[Any] = Field(None, description="The number of days for the renewal notice period, as a number only.")
    party_one: Optional[str] = Field(None, description="The full name of Party One (usually the Landlord). Keep the CASE as it is.")
    party_two: Optional[str] = Field(None, description="The full name of Party Two (usually the Tenant). Keep the CASE as it is.")


# --- Helper Functions for Text Extraction ---

def extract_text_from_docx(file_path):
    """Extract text from .docx files."""
    text = ""
    try:
        document = docx.Document(file_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        # TODO: Handle tables and other document elements as well
    except Exception as e:
        print(f"Error reading docx file {file_path}: {e}")
        return None
    return text

def extract_text_from_png(file_path):
    """Extract text from .png image files using EasyOCR."""
    # Initialize EasyOCR reader here so it's only loaded when needed
    try:
        # Initialize EasyOCR reader. Specify languages and model storage directory.
        # Removed download_cfg_path which caused the previous error.
        reader = easyocr.Reader(['en'], model_storage_directory='.')
    except Exception as e:
        print(f"Error initializing EasyOCR reader: {e}")
        print("Please ensure EasyOCR models are downloaded and internet is available for the first run.")
        return None # Return None if reader initialization fails

    text = ""
    try:
        # EasyOCR readtext returns a list of tuples: (bbox, text, prob)
        results = reader.readtext(file_path)
        # Concatenate the text content from the results
        for (bbox, text_content, prob) in results:
            text += text_content + "\n"
    except Exception as e:
        print(f"Error reading png file {file_path} with EasyOCR: {e}")
        return None
    return text


def extract_text_from_pdf(file_path):
    """Extract text from .pdf files using pypdf."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"Error reading pdf file {file_path}: {e}")
        return None
    return text


# --- Helper Function for Information Extraction using Groq ---

# Updated return type to directly return RentalAgreementInfo or None
def extract_info_with_groq(text_content: str) -> Optional[RentalAgreementInfo]:
    """
    Use Langchain with ChatGroq and with_structured_output to extract information
    and validate with Pydantic model.
    """
    global llm

    if not llm:
         print("Groq client not initialized (API key missing?). Skipping information extraction.")
         return None

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that extracts key information from rental agreements and formats the output as a JSON object with the following keys: agreement_value, agreement_start_date, agreement_end_date, renewal_notice_days, party_one, party_two. The output MUST be a valid JSON object matching the structure described."),
        ("human", "Extract the following information from the rental agreement text without keeping any information empty:\n"
                   "- The numerical value of the rent amount (integer only, no currency or commas).\n"
                   "- The start date of the lease agreement in DD.MM.YYYY format.\n"
                   "- The end date of the lease agreement in DD.MM.YYYY format.\n"
                   "- The number of days for the renewal notice period (number only).\n"
                   "- The full name of Party One (usually the Landlord). Maintain original casing(UPPER CASE OR LOWER CASE). Without Inverted Commas. Ignore Mr. Mrs. Dr. Sri. etc.\n"
                   "- The full name of Party Two (usually the Tenant). Maintain original casing(UPPER CASE OR LOWER CASE). Without Inverted Commas. Ignore Mr. Mrs. Dr. Sri. etc.\n\n"
                   "Text:\n{text_content}\n\n"
                   "Please provide the extracted information as a JSON object with the keys: agreement_value, agreement_start_date, agreement_end_date, renewal_notice_days, party_one, party_two."),
    ])

    try:
        structured_llm: Runnable[dict, RentalAgreementInfo] = llm.with_structured_output(RentalAgreementInfo)
        # Invoke the chained LLM
        extracted_data_model = structured_llm.invoke(prompt.format_messages(text_content=text_content))
        print("Successfully extracted information using with_structured_output.")
        return extracted_data_model # Returns the Pydantic model if successful
    except Exception as e:
        print(f"Error calling Groq API with structured output via Langchain: {e}")
        print(f"Problematic text content (might be long): {text_content[:500]}...")
        return None


# --- Main Extraction Logic ---

def get_base_filename_without_extensions(filename):
    """Removes all extensions from a filename."""
    basename = os.path.basename(filename)
    while True:
        filename_without_ext, ext = os.path.splitext(basename)
        if ext == '':
            break
        basename = filename_without_ext
    return basename


def process_single_file(file_path, output_csv_path, writer, csv_headers):
    """
    Processes a single file and writes extracted data to a CSV writer.
    Expects Pydantic model or None from extraction.
    """
    if not os.path.isfile(file_path):
        print(f"Error: Input path '{file_path}' is not a file.")
        return False

    full_filename = os.path.basename(file_path)
    filename_without_all_extensions = get_base_filename_without_extensions(full_filename)

    print(f"Processing file: {full_filename}")
    text_content = None
    lower_filename = full_filename.lower()

    # Determine file type and extract text
    if lower_filename.endswith('.docx') or lower_filename.endswith('.pdf.docx'):
        text_content = extract_text_from_docx(file_path)
    elif lower_filename.endswith('.png'):
        text_content = extract_text_from_png(file_path)
    elif lower_filename.endswith('.pdf'):
         text_content = extract_text_from_pdf(file_path)
    else:
        print(f"Skipping unsupported file type: {full_filename}")
        return False

    if text_content:
        # Optional: temporary print for debugging text content
        # print(f"--- Extracted text from {full_filename} ---")
        # print(text_content)
        # print("-------------------------------------------")

        # This function now returns a RentalAgreementInfo model or None
        extracted_data_model = extract_info_with_groq(text_content)

        if extracted_data_model: # Check if a valid model was returned
            # Use model_dump() to get a dictionary from the Pydantic model
            extracted_dict = extracted_data_model.model_dump()
            print(f"Using validated Pydantic model data for {full_filename}")

            # Prepare row data, getting values from the validated dictionary
            row_data = {"File Name": filename_without_all_extensions}
            row_data["Aggrement Value"] = extracted_dict.get("agreement_value", "N/A")
            row_data["Aggrement Start Date"] = extracted_dict.get("agreement_start_date", "N/A")
            row_data["Aggrement End Date"] = extracted_dict.get("agreement_end_date", "N/A")
            row_data["Renewal Notice (Days)"] = extracted_dict.get("renewal_notice_days", "N/A")
            row_data["Party One"] = extracted_dict.get("party_one", "N/A")
            row_data["Party Two"] = extracted_dict.get("party_two", "N/A")


            writer.writerow(row_data)
            print(f"Successfully processed and wrote data for {full_filename}")
            return True
        else:
            print(f"Could not extract information from {full_filename} (Groq structured output failed).")
            # We can still write a row with just the filename if extraction failed
            row_data = {"File Name": filename_without_all_extensions}
            # Fill other fields with N/A
            for header in csv_headers:
                if header != "File Name":
                    row_data[header] = "N/A"
            writer.writerow(row_data)
            print(f"Wrote row with default 'N/A' values for {full_filename} due to extraction failure.")
            return False # Indicate failure


    else:
        print(f"Could not extract text from {full_filename}")
        # Optionally write a row with just the filename if text extraction failed
        row_data = {"File Name": filename_without_all_extensions}
        for header in csv_headers:
            if header != "File Name":
                row_data[header] = "N/A"
        writer.writerow(row_data)
        print(f"Wrote row with default 'N/A' values for {full_filename} due to text extraction failure.")

        return False


# --- Script Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract information from a document or a directory of documents and write to a CSV file.')
    parser.add_argument('input_path', type=str, help='The path to the input document file or directory.')
    parser.add_argument('-o', '--output', type=str, default='extracted_data.csv', help='The path to the output CSV file.')
    args = parser.parse_args()

    input_path = args.input_path
    output_csv_path = args.output

    csv_headers = ["File Name", "Aggrement Value", "Aggrement Start Date", "Aggrement End Date", "Renewal Notice (Days)", "Party One", "Party Two"]
    write_headers = not os.path.exists(output_csv_path)

    try:
        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            if write_headers:
                writer.writeheader()

            if os.path.isfile(input_path):
                process_single_file(input_path, output_csv_path, writer, csv_headers)
            elif os.path.isdir(input_path):
                print(f"Processing directory: {input_path}")
                for filename in os.listdir(input_path):
                    file_path = os.path.join(input_path, filename)
                    if os.path.isfile(file_path):
                        process_single_file(file_path, output_csv_path, writer, csv_headers)
                    else:
                         print(f"Skipping non-file item in directory: {filename}")
            else:
                print(f"Error: Input path '{input_path}' is neither a file nor a directory.")

    except Exception as e:
        print(f"Error processing input or writing to CSV file {output_csv_path}: {e}")

# Note: The Flask app code provided previously in app.py is now superseded
# by the directory processing capability added to this script.