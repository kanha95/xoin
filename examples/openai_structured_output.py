"""Large nested structured extraction example (healthcare record).

Mirrors ``examples/native-js/openai-structured-output.js``.

The goal is to show **production-grade** nesting:

* enums via ``typing.Literal``
* lists of sub-models
* nullable insurance blocks
* explicit instructions so the model respects ISO dates + enum vocab

Environment variables
---------------------
``OPENAI_API_KEY``
    Required.
``OPENAI_MODEL``
    Optional chat model id.

Run::

    python examples/openai_structured_output.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from xoin import StructuredSpec, Xoin
from xoin.providers import OpenAIProvider


BloodGroup = Literal["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
Gender = Literal["male", "female", "other"]
Severity = Literal["low", "medium", "high"]
class PatientContact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    phone: str
    email: str | None = None
    address: str


class Patient(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    age: int = Field(ge=0, le=120)
    gender: Gender
    blood_group: BloodGroup = Field(alias="bloodGroup")
    contact: PatientContact


class MedicalHistoryEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    condition: str
    diagnosed_date: str = Field(alias="diagnosedDate")
    severity: Severity
    ongoing: bool


class Medication(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    dosage: str
    frequency: str
    start_date: str = Field(alias="startDate")
    end_date: str | None = Field(default=None, alias="endDate")


class LabResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    test_name: str = Field(alias="testName")
    result: str
    unit: str
    normal_range: str = Field(alias="normalRange")
    test_date: str = Field(alias="testDate")


class DoctorVisit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    doctor_name: str = Field(alias="doctorName")
    specialization: str
    visit_date: str = Field(alias="visitDate")
    notes: str | None = None
    prescription_given: bool = Field(alias="prescriptionGiven")


class Insurance(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    provider: str
    policy_number: str = Field(alias="policyNumber")
    valid_till: str = Field(alias="validTill")


class EmergencyContact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    relation: str
    phone: str


class HealthcareRecord(BaseModel):
    """Root schema returned by the LLM and validated locally."""

    model_config = ConfigDict(populate_by_name=True)

    patient: Patient
    medical_history: list[MedicalHistoryEntry] = Field(alias="medicalHistory")
    medications: list[Medication]
    allergies: list[str]
    lab_results: list[LabResult] = Field(alias="labResults")
    doctor_visits: list[DoctorVisit] = Field(alias="doctorVisits")
    insurance: Insurance | None = None
    emergency_contact: EmergencyContact = Field(alias="emergencyContact")


PROMPT = """
Generate a complete structured healthcare record.

IMPORTANT:
- Return ALL fields exactly as defined
- Use null if any value is missing
- Use ISO date format (YYYY-MM-DD)
- Use exact enum values for gender, severity, blood groups, and categories

Patient: Ruturaj Patil, 32, male, blood group O+
Lives in Pune, phone 9876543210, email ruturaj@gmail.com

Medical History:
- Diabetes diagnosed in 2020, medium severity, ongoing
- Asthma diagnosed in 2015, low severity, ongoing

Medications:
- Metformin 500mg twice daily since 2020
- Salbutamol inhaler as needed since 2015

Allergies:
- Penicillin
- Dust

Lab Results:
- HbA1c: 7.2%, normal 4-6%, tested Jan 2024
- Fasting Sugar: 130 mg/dL, normal 70-100, tested Jan 2024

Doctor Visits:
- Dr. Sharma (Endocrinologist) on Jan 2024, prescribed meds
- Dr. Mehta (Pulmonologist) on Feb 2024

Insurance:
- HDFC Ergo, policy HDF12345, valid till Dec 2026

Emergency Contact:
- Suresh Patil (Father), phone 9123456780
"""


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY.")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    async with Xoin(
        providers={
            "openai": OpenAIProvider(api_key=api_key, default_model=model),
        },
        default_provider="openai",
    ) as xoin:
        result = await xoin.generate(
            provider="openai",
            prompt=PROMPT.strip(),
            structured=StructuredSpec(
                response_model=HealthcareRecord,
                name="healthcare_record",
                description="Structured hospital intake + invoice-shaped totals.",
                mode="auto",
            ),
            temperature=0.2,
            max_tokens=2000,
        )

    # Preserve provider casing conventions from the JSON payload (camelCase aliases).
    payload = result.model_dump(mode="json", by_alias=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
