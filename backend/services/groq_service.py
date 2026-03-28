import os
import re
from groq import Groq
from backend.config import settings

# Initialize Groq client
# This requires GROQ_API_KEY to be in the environment!
try:
    client = Groq(api_key=settings.GROQ_API_KEY)
except Exception:
    client = None

def parse_cot(text):
    """
    Parses <reasoning> tags to extract step-by-step reasoning.
    Returns: list of dicts {"step_number": i, "content": ...}, and the "answer" string
    """
    # Try to find <reasoning> block
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', text, flags=re.DOTALL | re.IGNORECASE)
    
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()
        answer = text.replace(reasoning_match.group(0), "").strip()
    else:
        # Fallback if no specific tags
        parts = text.split("Final Answer:", 1)
        if len(parts) == 2:
            reasoning_text = parts[0].strip()
            answer = parts[1].strip()
        else:
            # Another common format
            if "\n\n" in text:
                parts = text.rsplit("\n\n", 1)
                reasoning_text = parts[0].strip()
                answer = parts[1].strip()
            else:
                reasoning_text = "Reasoning omitted or implicit."
                answer = text.strip()

    # Split reasoning by newlines or "Step:" to form discrete steps
    steps_raw = [s.strip() for s in reasoning_text.split('\n') if s.strip() and len(s.strip()) > 5]
    
    # Chunk logic if it didn't split well
    if len(steps_raw) == 1 and len(steps_raw[0]) > 100:
        steps_raw = [s.strip() + "." for s in steps_raw[0].split('. ') if s.strip()]
        
    formatted_steps = []
    
    # Deduplicate and limit
    for i, step in enumerate(steps_raw[:8]): 
        # Clean up step numbering if present
        cleaned_step = re.sub(r'^(Step|Phase)\s+\d+[:\.]?\s*', '', step, flags=re.IGNORECASE)
        cleaned_step = re.sub(r'^\d+[\.\)]\s*', '', cleaned_step)
        
        formatted_steps.append({
            "step_number": i + 1,
            "content": cleaned_step,
            "is_missing": False,
            "is_improved": False
        })
        
    if not answer and len(formatted_steps) > 0:
        answer = formatted_steps[-1]["content"]
        formatted_steps = formatted_steps[:-1]
        
    return formatted_steps, answer

def generate_teacher_cot(question: str):
    if not client: return [{"step_number": 1, "content": "Groq API key missing"}], "Configuration Error", ""
    
    prompt = f"""
    Answer the following question. You MUST provide detailed, step-by-step reasoning enclosed in <reasoning> tags before your final answer.
    Break down the logic clearly. 
    
    Question: {question}
    """
    
    response = client.chat.completions.create(
        model=settings.TEACHER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2048
    )
    
    output = response.choices[0].message.content
    steps, answer = parse_cot(output)
    return steps, answer, output

def generate_student_direct(question: str):
    if not client: return "Configuration Error"
    
    prompt = f"""
    Answer the following question directly and concisely without showing your work or reasoning. Just provide the final answer.
    
    Question: {question}
    """
    
    response = client.chat.completions.create(
        model=settings.STUDENT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512
    )
    
    return response.choices[0].message.content

def dict_to_steps(steps_dict):
    """Helper to convert dictionary to steps structure if needed"""
    return steps_dict

def generate_student_distilled(question: str, teacher_steps: list):
    if not client: return [{"step_number": 1, "content": "Groq API key missing"}], "Configuration Error", ""
    
    # Limit guide to prevent context stuffing
    reasoning_guide = "\n".join([f"Step {s['step_number']}: {s['content']}" for s in teacher_steps[:5]])
    
    prompt = f"""
    You are a smaller student model learning from a teacher. Answer the question using the following reasoning structure as a guide:
    
    Guiding Logic:
    {reasoning_guide}
    
    Question: {question}
    
    Provide your reasoning briefly in <reasoning> tags, then your final answer. It is okay to skip steps or compress the logic.
    """
    
    response = client.chat.completions.create(
        model=settings.STUDENT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4, # Slightly higher temp to simulate imperfection
        max_tokens=1024
    )
    
    output = response.choices[0].message.content
    steps, answer = parse_cot(output)
    
    # Simulate distillation imperfection visually
    if len(steps) > 2 and len(teacher_steps) > 3:
        # Mark a step as missing/condensed to show loss of fidelity
        steps[min(1, len(steps)-1)]["is_missing"] = True
        
    return steps, answer, output
    
def generate_experiment(question: str, temperature: float, top_k: int, top_p: float, cot: bool):
    if not client: return "API error"
    
    prompt = f"Question: {question}"
    if cot:
        prompt = f"Answer step-by-step using <reasoning> tags.\n{prompt}"
        
    response = client.chat.completions.create(
        model=settings.STUDENT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=1024
    )
    return response.choices[0].message.content
