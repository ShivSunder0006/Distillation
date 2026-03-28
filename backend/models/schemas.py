from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class AskRequest(BaseModel):
    question: str

class StepInfo(BaseModel):
    step_number: int
    content: str
    is_missing: Optional[bool] = False
    is_improved: Optional[bool] = False

class GenerationResponse(BaseModel):
    teacher_cot: List[StepInfo]
    teacher_answer: str
    student_no_cot: str
    student_cot: List[StepInfo]
    student_cot_answer: str
    processing_time_ms: int

class RAGUploadResponse(BaseModel):
    status: str
    chunks_processed: int
    filename: str

class RAGQueryRequest(BaseModel):
    question: str
    
class RAGChunk(BaseModel):
    content: str
    score: float
    source: str

class RAGResponse(BaseModel):
    answer: str
    retrieved_chunks: List[RAGChunk]

class ExperimentRequest(BaseModel):
    question: str
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    cot_enabled: bool = True

class DistillationResponse(BaseModel):
    alignment_score: float
    teacher_steps: List[StepInfo]
    student_steps: List[StepInfo]
