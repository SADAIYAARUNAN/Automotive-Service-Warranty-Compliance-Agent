from typing import Dict, Any
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


# Import State Schema
from src.state import AgentState


# Initalize presidio engine
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


# Custom Recognizser for Vehicle Indentification Number (VIN)
vin_pattern = Pattern(
    name="vin_pattern",
    regex=r"[A-HJ-NPR-Z0-9]{17}",
    score=0.9
)


vin_recognizer = PatternRecognizer(
    supported_entity="VIN",
    patterns=[vin_pattern]
)


analyzer.registry.add_recognizer(vin_recognizer)


def sanitize_data(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node : sanitize_data
    Reads raw_service_report, Scrubs the Standard PII and VIN's,
    ant outputs the sanitized data.
    """

    raw_data = state.raw_service_report

    if not raw_data:
        return{"sanitized_report": "No raw data provided."}



    # Analyze the text for PII and VIN
    results = analyzer.analyze(
        text = raw_data,
        entities=["PERSON", "LOCATION", "PHONE_NUMBER", "EMAIL_ADDRESS", "VIN"],
        language="en"
    )


    # Detect the specific entities and anonymize with mask
    anonymized_result = anonymizer.anonymize(
        text=raw_data,
        analyzer_results=results,
        operators= {
             "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
             "PERSON": OperatorConfig("replace", {"new_value": "<PERSON_REDACTED>"}),
             "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION_REDACTED>"}),
             "PHONE_NUMBER": OperatorConfig("replace", {"new_value":"<PHONE_REDCATED>"}),
             "VIN": OperatorConfig("replace",{"new_value":"<VIN_REDACTED>"})
        }
    )

    # Return the updated state
    return {"sanitized_report": anonymized_result.text}


if __name__ == "__main__":

    mock_data = AgentState(
          raw_service_report = """
          Service Ticket #SRV-99421-EU. Customer Sarah Van Dijk dropped off her vehicle at Amsterdam Central Motors located at Keizersgracht 424, 
          1016 GC Amsterdam. Contact phone: +31-20-555-0199. Vehicle VIN: WBA5SER7890123456, License Plate: 45-HZ-KJ. 
          Customer complaint: Infotainment screen went completely blank during driving. Mechanic initial notes: Technician inspected the dashboard wiring harness and discovered 
          aftermarket third-party amplifier spliced directly into the main CAN bus data lines. Customer stated an independent garage installed custom audio equipment last month. 
          Requesting full warranty coverage replacement for the central display unit and infotainment module.

        """
    )


    result = sanitize_data(mock_data)

    print("RAW DATA")
    print(mock_data.raw_service_report.strip())
    print("\n---Sanitized data---")
    print(result["sanitized_report"].strip())