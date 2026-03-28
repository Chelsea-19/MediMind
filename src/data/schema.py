from pydantic import BaseModel, Field
from typing import List, Optional

class MedicalResponse(BaseModel):
    summary: str = Field(description="Brief summary of the response.")
    reasoning_basis: List[str] = Field(description="Evidence snippets used to form the response.")
    risk_level: str = Field(description="Risk level of the patient's query/symptoms (e.g., Routine, Urgent, Emergency).")
    recommended_action: str = Field(description="Next steps recommended for the patient.")
    uncertainty_note: Optional[str] = Field(description="Note on the limits of this AI's knowledge or ability.")
    disclaimer: str = Field(default="本系统提供的信息仅供初步健康教育和参考，不能代替专业医生的诊断和治疗。如遇紧急情况请立即就医。", description="Medical disclaimer.")

class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: dict
