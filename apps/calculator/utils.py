import google.generativeai as genai
import ast
import json
import re
from PIL import Image
from constants import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

def clean_response_text(text: str) -> str:
    """
    Cleans Gemini response by:
    - Removing markdown formatting (```...```)
    - Extracting content between brackets if possible
    """
    text = text.strip()

    # Remove markdown-style code block
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```"):
            text = "\n".join(lines[1:-1])
        else:
            text = text.strip("`").strip()

    # Replace smart quotes with standard quotes
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Extract content inside a list if present
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    return match.group(0) if match else text.strip()



def analyze_image(img: Image, dict_of_vars: dict):
    img = img.resize((min(img.width, 512), min(img.height, 512)))
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    dict_of_vars_str = json.dumps(dict_of_vars, ensure_ascii=False)
    
    prompt = (
        f"If the expression includes variables (like x, y), substitute their values from the user-assigned dictionary before solving. "
        f"For example, if x=5 and y=10, then 2x+y = 2*5 + 10 = 20. "
        f"Make sure to interpret expressions like 2x as 2 * x."
        f"You have been given an image with some mathematical expressions, equations, or graphical problems, and you need to solve them. "
        f"Note: Use the PEMDAS rule for solving mathematical expressions. PEMDAS stands for the Priority Order: Parentheses, Exponents, Multiplication and Division (from left to right), Addition and Subtraction (from left to right). "
        f"For example: "
        f"Q. 2 + 3 * 4 -> (3 * 4) => 12, 2 + 12 = 14. "
        f"Q. 2 + 3 + 5 * 4 - 8 / 2 -> 5 * 4 => 20, 8 / 2 => 4, 2 + 3 => 5, 5 + 20 => 25, 25 - 4 => 21. "
        f"YOU CAN HAVE FIVE TYPES OF EXPRESSIONS IN THIS IMAGE: "
        f"1. Simple math like 2 + 2: Return as a LIST of one dict: [{{'expr': '2 + 2', 'result': 4}}] "
        f"2. Set of Equations: solve and return list of dicts with 'assign': True for each variable. "
        f"3. Variable Assignments like x = 5: return [{{'expr': 'x', 'result': 5, 'assign': True}}] "
        f"4. Graphical math problems or scenes: describe them, then return [{{'expr': description, 'result': answer}}] "
        f"5. Abstract concepts (e.g. patriotism, love): return [{{'expr': description, 'result': concept}}] "
        f"Use this dictionary of known values if needed: {dict_of_vars_str} "
        f"DO NOT wrap the output in code blocks. ONLY return a valid Python or JSON-style list of dicts. No markdown, no commentary. "
    )

    response = model.generate_content([prompt, img])
    print("Raw Gemini response:\n", response.text)

    cleaned_text = clean_response_text(response.text)
    print("Cleaned Gemini response:\n", cleaned_text)

    answers = []
    try:
        answers = ast.literal_eval(cleaned_text)
    except Exception as e:
        print(f"Literal eval failed: {e}")
        try:
            answers = json.loads(cleaned_text)
        except Exception as json_error:
            print(f"JSON parse failed: {json_error}")
            answers = []

    print('returned answer ', answers)

  
    for answer in answers:
        answer['assign'] = answer.get('assign', False)

    return answers
