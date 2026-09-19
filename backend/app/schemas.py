from typing import Optional, List
from pydantic import BaseModel, Field


class FieldEvidence(BaseModel):
    field: str
    quote: str = ""
    page: int
    region: int


class LicenceExtraction(BaseModel):
    full_name: Optional[str] = None
    licence_number: Optional[str] = None
    date_of_birth: Optional[str] = None
    date_of_issue: Optional[str] = None
    date_of_expiry: Optional[str] = None
    address: Optional[str] = None
    vehicle_classes: List[str] = Field(default_factory=list)
    issuing_authority: Optional[str] = None
    blood_group: Optional[str] = None
    father_spouse_name: Optional[str] = None
    restrictions: Optional[str] = None
    other_information: List[str] = Field(default_factory=list)
    field_evidence: List[FieldEvidence] = Field(default_factory=list)


class OCRLine(BaseModel):
    text: str
    confidence: float
    bbox: list = []


class LicenceDocument(BaseModel):
    id: str
    page: int
    region: int
    title: str
    preview_data_url: str
    ocr_text: str
    ocr_lines: list[OCRLine] = []
    extraction: LicenceExtraction


class AnalyzeResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    preview_data_url: str
    documents: List[LicenceDocument]


class SaveRequest(BaseModel):
    documents: List[LicenceDocument]


class ChatRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=2, max_length=1000)
    selected_region: Optional[int] = None


class ChatSource(BaseModel):
    page: int
    region: int
    chunk_id: str
    text: str


class ChatResponse(BaseModel):
    answer: str
    found: bool
    sources: List[ChatSource] = Field(default_factory=list)
