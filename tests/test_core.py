import ast
import re
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from rfp_intel.agents.extract import reset_search_override, set_search_override
from rfp_intel.agents.fields import ALL_FIELDS, SPECIALIST_FIELDS
from rfp_intel.agents.graph import build_extract_graph
from rfp_intel.agents.llm import reset_llm_override, set_llm_override
from rfp_intel.agents.reconcile import reconcile_candidates
from rfp_intel.agents.schemas import AnswerResult, CitedSource, ExtractedItem, SectionExtraction, SummaryResult
from rfp_intel.agents.validate import judge
from rfp_intel.ingestion.browse import _display_status, step_rows
from rfp_intel.ingestion.parser import page_has_extractable_text, text_page_ranges
from rfp_intel.ingestion.classify import classify_document
from rfp_intel.ingestion.diff import diff_hashes
from rfp_intel.ingestion.sections import RawItem, build_sections
from rfp_intel.observability.trace import bind_trace, reset_trace
from rfp_intel.resilience import NonRetryableError, backoff_delay, is_retryable, retry_with_backoff
from rfp_intel.schemas import FieldValue, SearchHit
from rfp_intel.search.chunking import chunk_sections
from rfp_intel.search.expand import expand_query
from rfp_intel.search.fusion import reciprocal_rank_fusion
from metrics import mean, recall_at_k, reciprocal_rank


def test_ocr_ranges_follow_pages_without_text():
    assert page_has_extractable_text("  \n") is False
    assert page_has_extractable_text("Bid No. 12") is True
    assert text_page_ranges([True, True, False, False, True]) == [
        (1, 2, True),
        (3, 4, False),
        (5, 5, True),
    ]
    assert text_page_ranges([]) == []


def test_pipeline_status_follows_the_tables():
    assert _display_status(None, 0, 0, 0) == "not_started"
    assert _display_status("parsing", 0, 0, 0) == "parsing"
    assert _display_status("parsed", 2, 0, 0) == "parsed"
    assert _display_status("parsed", 2, 4, 1) == "chunked"
    assert _display_status("chunked", 2, 4, 4) == "ready"
    assert _display_status("parsed", 0, 0, 0, failed_files=1) == "failed"
    with pytest.raises(ValueError):
        step_rows("Bid 1", "jobs")


def test_classify_document_types():
    assert classify_document(Path("page.html")) == ("bid_page", None)
    assert classify_document(Path("Addendum 2 RFP.pdf")) == ("addendum", 2)
    assert classify_document(Path("Contract_Affidavit.pdf")) == ("affidavit", None)
    assert classify_document(Path("Dell_Laptop_Specs.pdf")) == ("specs", None)
    assert classify_document(Path("PORFP_-_Dell_Laptop_Final.pdf")) == ("rfp", None)


def test_diff_hashes_only_new_and_changed_files():
    diff = diff_hashes(
        {"a.pdf": "1", "b.pdf": "9", "c.pdf": "3"},
        {"a.pdf": "1", "b.pdf": "2"},
    )
    assert diff.added == ["c.pdf"]
    assert diff.changed == ["b.pdf"]
    assert diff.removed == []
    assert diff_hashes({"a.pdf": "1"}, {"a.pdf": "1"}).has_changes is False


def test_sections_keep_heading_and_table():
    sections = build_sections(
        [
            RawItem("heading", "Closing Date", level=2, page=1),
            RawItem("text", "Bids are due March 1, 2026.", level=2, page=1),
            RawItem(
                "table",
                "",
                level=2,
                page=2,
                table_markdown="| Model | RAM |\n| --- | --- |\n| Latitude 3450 | 16 GB |",
                table_rows=[{"Model": "Latitude 3450", "RAM": "16 GB"}],
            ),
        ]
    )
    assert sections[0].heading_path == ["Closing Date"]
    assert "March 1, 2026" in sections[0].body_text
    assert "Latitude 3450 16 GB" in sections[0].body_text
    assert "|" not in sections[0].body_text
    assert sections[0].tables == []


def test_table_without_rows_stays_out_of_the_section_body():
    sections = build_sections(
        [
            RawItem("heading", "Lines", level=1, page=3),
            RawItem(
                "table",
                "",
                page=3,
                table_markdown="| Model |\n| --- |\n| M1 |",
                table_rows=[],
            ),
        ]
    )
    assert sections[0].body_text == ""
    assert sections[0].tables[0].markdown.startswith("| Model |")


def test_chunk_metadata_stays_inside_section():
    sections = build_sections(
        [
            RawItem("heading", "Specifications", level=1, page=4),
            RawItem("text", "word " * 30, level=1, page=4),
            RawItem("heading", "Bond", level=1, page=5),
            RawItem("text", "A bid bond is not required.", level=1, page=5),
        ]
    )
    chunks = chunk_sections(sections, max_tokens=20)
    spec_chunks = [chunk for chunk in chunks if chunk.heading_path == ["Specifications"]]
    bond_chunks = [chunk for chunk in chunks if chunk.heading_path == ["Bond"]]
    assert spec_chunks
    assert bond_chunks
    assert all(chunk.page_number == 4 for chunk in spec_chunks)
    assert all("bid bond" not in chunk.text.lower() for chunk in spec_chunks)
    assert bond_chunks[0].page_number == 5
    assert "Specifications" in spec_chunks[0].text


def test_table_chunks_repeat_header():
    sections = build_sections(
        [
            RawItem("heading", "Lines", level=1, page=3),
            RawItem(
                "table",
                "",
                page=3,
                table_markdown="| Model |\n| --- |\n" + "\n".join(f"| M{i} |" for i in range(10)),
                table_rows=[],
            ),
        ]
    )
    chunks = chunk_sections(sections, max_tokens=8)
    table_chunks = [chunk for chunk in chunks if "| Model |" in chunk.text or "Model" in chunk.text]
    assert len(table_chunks) >= 2
    assert all("Model" in chunk.text for chunk in table_chunks)


def test_rrf_prefers_documents_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert fused[0][0] == "b"


def test_expand_query_adds_deadline_synonyms():
    expanded = expand_query("What is the deadline?")
    assert "closing date" in expanded
    assert expanded.startswith("What is the deadline?")


def test_backoff_is_bounded_and_grows_in_ceiling():
    assert backoff_delay(0, base=0.5, cap=8) <= 0.5
    assert backoff_delay(10, base=0.5, cap=8) <= 8


def test_retry_skips_auth_errors_and_backs_off():
    delays = []

    def fail():
        error = RuntimeError("nope")
        error.status_code = 503
        raise error

    with pytest.raises(RuntimeError):
        retry_with_backoff(fail, max_attempts=3, base=0.5, cap=8, on_retry=lambda attempt, delay, exc: delays.append((attempt, delay)), sleeper=lambda seconds: None)
    assert len(delays) == 2
    assert delays[1][0] == 1

    def denied():
        raise NonRetryableError("bad key")

    with pytest.raises(NonRetryableError):
        retry_with_backoff(denied, max_attempts=4, sleeper=lambda seconds: None)
    assert is_retryable(NonRetryableError("x")) is False


def test_field_value_requires_a_citation():
    with pytest.raises(ValidationError):
        FieldValue(value="JA-207652", sources=[], confidence=0.9)
    missing = FieldValue(value=None, sources=[])
    assert missing.notes == "Not found in documents"


def test_validator_rejects_unsupported_short_value():
    assert judge("Bid Number", "JA-207652", "Solicitation Number JA-207652") == "pass"
    assert judge("Bid Number", "NOPE", "Solicitation Number JA-207652") == "fail"
    assert judge("Due Date", "soon", "due soon") == "fail"
    assert judge("Bid Bond Requirement", None, "text") == "not_found"


def test_addendum_keeps_the_latest_value():
    chosen, changes = reconcile_candidates(
        [
            {"field_name": "Due Date", "value": "March 1", "addendum_number": 0, "confidence": 0.8, "file_name": "rfp.pdf", "page": 1, "notes": ""},
            {"field_name": "Due Date", "value": "March 15", "addendum_number": 2, "confidence": 0.9, "file_name": "addendum2.pdf", "page": 1, "notes": ""},
        ],
        ["Due Date"],
    )
    assert chosen["Due Date"]["value"] == "March 15"
    assert changes[0]["previous_value"] == "March 1"
    assert "addendum 2" in chosen["Due Date"]["notes"].lower()


CHUNK = (
    "Solicitation Number JA-207652. Title Student and Staff Computing Devices. "
    "Issuing organization Dallas Independent School District. "
    "Contact Procurement Services 972-925-3700 ProcurementCS@dallasisd.org. "
    "Closing Date 07/09/2024 03:00 PM EDT. Prebid Conference 06/10/2024 03:00 PM EDT. "
    "Electronic submission through the portal. Term of bid is five years. "
    "Delivery window is 30 days after award. Installation imaging is required. "
    "Bid bond is not required. Payment terms are Net 30. "
    "Additional documentation includes a W-9. Manufacturer registration is Dell. "
    "Cooperative contract is not used. Model Latitude 3450. Part number ABC-123. "
    "Product student laptop quantity 100. "
    "Product specification requires 16 GB RAM and a three year warranty."
)

CATALOG = {
    "Bid Number": "JA-207652",
    "Title": "Student and Staff Computing Devices",
    "company_name": "Dallas Independent School District",
    "contact_info": "ProcurementCS@dallasisd.org",
    "Due Date": "07/09/2024 03:00 PM EDT",
    "Bid Submission Type": "Electronic submission through the portal",
    "Pre Bid Meeting": "06/10/2024 03:00 PM EDT",
    "Delivery Date": "30 days after award",
    "Installation": "Installation imaging is required",
    "Term of Bid": "five years",
    "Bid Bond Requirement": "Bid bond is not required",
    "Payment Terms": "Net 30",
    "Any Additional Documentation Required": "Additional documentation includes a W-9",
    "MFG for Registration": "Dell",
    "Contract or Cooperative to use": "not used",
    "Model_no": "Latitude 3450",
    "Part_no": "ABC-123",
    "Product": "student laptop quantity 100",
    "Product Specification": "Product specification requires 16 GB RAM and a three year warranty",
}


def _fake_search(query, bid_id=None, top_k=8, doc_type=None):
    text = CHUNK
    value = "JA-207652"
    if "Previous attempt was rejected" in query:
        text = CHUNK + " Confirmed solicitation JA-207652."
        value = "JA-207652"
    hit = SearchHit(
        chunk_id="c1",
        text=text,
        score=0.9,
        file_name="bid.html",
        page_number=1,
        section_heading="Basic Information",
        section_id="s1",
        bid_id=bid_id or "Bid 1",
        doc_type="bid_page",
        addendum_number=0,
    )
    return [hit]


def _fake_llm(schema, system, user):
    if schema is SectionExtraction:
        match = re.search(r"Fields: (\[.*?\])", user)
        requested = ast.literal_eval(match.group(1)) if match else []
        items = []
        for name in requested:
            value = CATALOG[name]
            if "Previous attempt was rejected" not in user and name == "Bid Number" and "force-fail" in user:
                value = "NOT-IN-DOC"
            items.append(ExtractedItem(field_name=name, value=value, confidence=0.91, notes="", chunk_id="c1"))
        return SectionExtraction(fields=items)
    if schema is SummaryResult:
        return SummaryResult(
            summary=(
                "Dallas Independent School District is soliciting student and staff computing devices under JA-207652. "
                "Proposals are due 07/09/2024 03:00 PM EDT. The term is five years. "
                "A student laptop, model Latitude 3450, is requested with 16 GB RAM."
            ),
            chunk_ids=["c1"],
        )
    if schema is AnswerResult:
        return AnswerResult(answer="JA-207652", citations=[CitedSource(file="bid.html", page=1, section="Basic Information")])
    raise AssertionError(schema)


def test_extract_graph_cites_every_value(monkeypatch):
    monkeypatch.setattr("rfp_intel.agents.graph.sleep", lambda seconds: None)
    sink = []
    tokens = bind_trace("memory", sink)
    llm_token = set_llm_override(_fake_llm)
    search_token = set_search_override(_fake_search)
    try:
        final = build_extract_graph().invoke({"bid_folder": "Bid 1"})
    finally:
        reset_trace(tokens)
        reset_llm_override(llm_token)
        reset_search_override(search_token)
    record = final["record"]
    assert set(record["fields"]) == set(ALL_FIELDS)
    for name in SPECIALIST_FIELDS:
        field = record["fields"][name]
        assert field["value"]
        assert field["sources"]
        assert field["sources"][0]["file"] == "bid.html"
    assert record["fields"]["Bid Summary"]["value"]
    assert any(step["agent"] == "orchestrator" for step in sink)
    assert any(step["agent"] == "addendum" for step in sink)
    assert any(step["agent"] == "validator" for step in sink)
    example = ROOT / "examples" / "traces" / "example_extract_trace.json"
    example.parent.mkdir(parents=True, exist_ok=True)
    example.write_text(__import__("json").dumps(sink, indent=2), encoding="utf-8")


def test_recall_and_mrr():
    question = {"passage": "JA-207652", "file_contains": "bid", "bid_id": "Bid 1"}
    hits = [
        {"text": "nothing", "file_name": "bid.html", "bid_id": "Bid 1"},
        {"text": "Solicitation JA-207652", "file_name": "bid.html", "bid_id": "Bid 1"},
    ]
    assert recall_at_k(hits, question, 5) == 1.0
    assert reciprocal_rank(hits, question) == 0.5
    assert mean([1.0, 0.0]) == 0.5


def test_sql_migrations_split_on_statement_boundaries():
    from rfp_intel.db.migrate import MIGRATIONS_DIR, split_sql

    script = (MIGRATIONS_DIR / "001_initial.sql").read_text(encoding="utf-8")
    statements = split_sql(script)
    joined = "\n".join(statements)
    for table in (
        "bids",
        "source_files",
        "sections",
        "tables",
        "chunks",
        "ingestion_jobs",
        "agent_runs",
        "agent_steps",
        "extracted_fields",
        "addendum_changes",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in joined
    assert "to_tsvector" in joined
    quoted = split_sql("INSERT INTO t (note) VALUES ('a;b'); SELECT 1;")
    assert quoted == ["INSERT INTO t (note) VALUES ('a;b')", "SELECT 1"]


def test_docling_html_sections_and_table():
    pytest.importorskip("docling")
    from rfp_intel.ingestion.parser import parse_file

    parsed = parse_file(ROOT / "tests" / "fixtures" / "sample_bid.html")
    text = "\n".join(section.body_text for section in parsed.sections)
    assert "March 1, 2026" in text or "JA-100" in text
    tables = [table for section in parsed.sections for table in section.tables]
    assert tables
    assert "Latitude 3450" in tables[0].markdown or any("Latitude" in str(row) for row in tables[0].rows)
