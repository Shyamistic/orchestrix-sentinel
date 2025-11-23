from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class EmployeeRequest(BaseModel):
employee_id: str
request_text: str
department: Optional[str] = None
metadata: Optional[Dict[str, Any]] = None

class SkillExecution(BaseModel):
skill_name: str
inputs: Dict[str, Any]
timestamp: Optional[str] = None

class ApprovalDecision(BaseModel):
workflow_id: str
approver: str
decision: str # APPROVE / REJECT
notes: Optional[str] = None

class RegisterPlugin(BaseModel):
name: str
entry_point: str
version: str = "1.0.0"
manifest: Optional[Dict[str, Any]] = None