import google.generativeai as genai
import dotenv
from PIL import Image
import os
import asyncio

dotenv.load_dotenv()

def initialize_api(model_name = "gemini-2.0-flash-lite"):
    """
    Initialize the Gemini API with the provided API key.
    """
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model_name)
    
    return model      


model = initialize_api()




def get_response_google(prompt:list[str, Image]):
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
    image = Image.open("data\epickitchen\P01_101_frame_0000000188.jpg")
    prompt = "what is the location ? Return only the answer"
    prompt = "what is the object in hands? Return only the answer if hands are present otherwise return no hands"
    response = get_response([prompt, image])
    print(response)    