"""The 20 bid fields, grouped so specialists can run in parallel."""

FIELD_GROUPS: dict[str, list[str]] = {
    "identity": ["Bid Number", "Title", "company_name", "contact_info"],
    "logistics": [
        "Due Date",
        "Bid Submission Type",
        "Pre Bid Meeting",
        "Delivery Date",
        "Installation",
    ],
    "commercial": [
        "Term of Bid",
        "Bid Bond Requirement",
        "Payment Terms",
        "Any Additional Documentation Required",
        "MFG for Registration",
        "Contract or Cooperative to use",
    ],
    "product": ["Model_no", "Part_no", "Product", "Product Specification"],
}

SPECIALIST_FIELDS: list[str] = [field for fields in FIELD_GROUPS.values() for field in fields]
SUMMARY_FIELD = "Bid Summary"
ALL_FIELDS: list[str] = SPECIALIST_FIELDS + [SUMMARY_FIELD]

DATE_FIELDS = {"Due Date", "Pre Bid Meeting", "Delivery Date"}
