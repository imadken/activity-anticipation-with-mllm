import google.generativeai as genai
import dotenv
from PIL import Image
import os
from typing import Optional
from time import time


dotenv.load_dotenv()

def initialize_api(model_name = "gemini-2.0-flash-lite"):
    """
    Initialize the Gemini API with the provided API key.
    """
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model_name)
    
    return model      


model = initialize_api()




def get_response_google(prompt:list[str, Optional[Image.Image]]):
    """
    Get a response from the Gemini API for a given prompt.
    """
    try:
        response = model.generate_content(contents = prompt)
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return None
    
if __name__ == "__main__":
    # Example usage
    SYSTEM_PROMPT = (
    "You are an expert in building knwowledge graphs.\n"
    "Given an object, you will return the most common affordances that can be done with it while in hands.\n"
    "Return the answer in the following format separated by commas:\n"
    "affordance1, affordance2, ...\n"
    "Return only the answer, do not include any other text or explanation and keep the answers simple (only one word or verb if possible).\n"
)
    image = Image.open("data\epickitchen\P01_101_frame_0000000188.jpg")
    prompt = "what is the location ? Return only the answer"
    prompt = "what is the object in hands? Return only the answer if hands are present otherwise return no hands"
    
    
    objects = ["coffee", "pan", "fork", "knife", "cup","coffee", "pan", "fork", "knife", "cup"]
    total_time = 0
    for obj in objects:
        
        Start = time()
        prompt = f"{SYSTEM_PROMPT} what are the affordances of  {obj} ?"
        response = get_response_google([prompt])
        end = time() 
        print(f"Response time: {end - Start:.2f} seconds")
        total_time += (end - Start)
        print(f"Object: {obj}, Response: {response}")
        
    print(f"average time for llm call per object: {total_time/5:.2f} seconds")