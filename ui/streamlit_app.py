"""Streamlit UI. Each bid is parsed, chunked, and embedded only when the user starts that step."""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

API = os.getenv("API_BASE_URL").rstrip("/")


def _decode(response: httpx.Response):
    if response.is_error:
        detail = response.text
        try:
            body = response.json()
            detail = body.get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(detail)
    return response.json()


def api_get(path: str, params: dict | None = None):
    return _decode(httpx.get(f"{API}{path}", params=params, timeout=60))


def api_post(path: str, payload: dict, timeout: float = 600):
    return _decode(httpx.post(f"{API}{path}", json=payload, timeout=timeout))


st.set_page_config(page_title="RFP Intelligence", layout="wide")
st.title("RFP Intelligence Platform")


def load_status() -> dict | None:
    try:
        return api_get("/pipeline/status")
    except Exception as exc:
        st.error(f"API is not reachable at {API}. Start it with `python main.py serve`. ({exc})")
        return None


status = load_status()
retrieval_ready = bool(status and status.get("retrieval_ready"))
page = st.sidebar.radio("Page", ["Pipeline", "Search", "Ask", "Extract"])
if status is not None:
    points = status.get("qdrant_points") or 0
    if retrieval_ready:
        st.sidebar.success(f"Qdrant has {points} point{'s' if points != 1 else ''}.")
    else:
        st.sidebar.warning("Search, Ask, and Extract stay off until a bid is embedded.")

if page == "Pipeline":
    st.subheader("Bid pipeline")
    st.caption("Nothing runs on its own. Pick a bid folder, then parse, chunk, and embed it. Each step shows the rows it wrote.")
    if st.button("Refresh"):
        st.rerun()
    if status is None:
        st.stop()
    bids = status.get("bids") or []
    if not bids:
        st.info("No bid folders under the data directory yet. Add a folder of PDF or HTML files, then refresh.")
        st.stop()
    st.dataframe(
        [
            {
                "bid": row["folder_name"],
                "status": row["status"],
                "files": row["files"],
                "parsed": row["parsed_files"],
                "failed": row["failed_files"],
                "sections": row["sections"],
                "tables": row["tables"],
                "chunks": row["chunks"],
                "embedded": row["embedded"],
            }
            for row in bids
        ],
        use_container_width=True,
        hide_index=True,
    )
    selected = st.selectbox("Bid", [row["folder_name"] for row in bids])
    current = next(row for row in bids if row["folder_name"] == selected)
    parse_col, chunk_col, embed_col = st.columns(3)
    with parse_col:
        st.markdown("**1. Parse**")
        st.caption("Writes source files, sections, and tables.")
        if st.button("Parse", type="primary", key="parse"):
            with st.spinner(f"Parsing {selected}..."):
                try:
                    api_post("/pipeline/parse", {"bid_id": selected})
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.rerun()
    with chunk_col:
        st.markdown("**2. Chunk**")
        st.caption("Writes chunks from the parsed documents.")
        if st.button("Chunk", disabled=not current["can_chunk"], key="chunk"):
            with st.spinner(f"Chunking {selected}..."):
                try:
                    api_post("/pipeline/chunk", {"bid_id": selected})
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.rerun()
    with embed_col:
        st.markdown("**3. Embed**")
        st.caption("Writes vectors to Qdrant.")
        if st.button("Embed", disabled=not current["can_embed"], key="embed"):
            with st.spinner(f"Embedding {selected}..."):
                try:
                    api_post("/pipeline/embed", {"bid_id": selected})
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.rerun()
    if current["chunks"] and not current["can_embed"]:
        st.caption("Every chunk for this bid already has a Qdrant point. Parse again only if files changed, then chunk and embed.")
    view = st.radio(
        "Table",
        ["Files", "Sections", "Tables", "Chunks"],
        horizontal=True,
    )
    step = view.lower()
    params = {"bid_id": selected, "step": step}
    if view != "Files":
        try:
            file_payload = api_get("/pipeline/rows", {"bid_id": selected, "step": "files"})
        except Exception as exc:
            st.error(str(exc))
            file_payload = {"rows": []}
        file_names = [row.get("relative_path") for row in file_payload.get("rows") or [] if row.get("relative_path")]
        chosen = st.selectbox("File", ["All files", *file_names])
        if chosen != "All files":
            params["file"] = chosen
    try:
        payload = api_get("/pipeline/rows", params)
    except Exception as exc:
        st.error(str(exc))
    else:
        rows = payload.get("rows") or []
        if not rows:
            if params.get("file"):
                st.info(f"No {view.lower()} rows for {params['file']}.")
            else:
                st.info(f"No {view.lower()} rows for {selected} yet. Run the step that fills this table.")
        else:
            st.caption(f"{len(rows)} row{'s' if len(rows) != 1 else ''} shown. Long text is truncated.")
            st.dataframe(rows, use_container_width=True, hide_index=True)

elif not retrieval_ready:
    st.subheader(page)
    st.warning(
        "This page uses the embedded chunks in Qdrant. "
        "Open Pipeline, then parse, chunk, and embed a bid. Search, Ask, and Extract turn on after that."
    )
    st.stop()

elif page == "Search":
    st.subheader("Hybrid search")
    query = st.text_input("Query", placeholder="closing date for student devices")
    col1, col2, col3 = st.columns(3)
    bid_id = col1.text_input("Bid folder", placeholder="Bid 1")
    doc_type = col2.selectbox("Document type", ["", "bid_page", "rfp", "addendum", "specs", "affidavit"])
    top_k = col3.number_input("Top k", min_value=1, max_value=20, value=5)
    if st.button("Search", type="primary") and query.strip():
        params = {"q": query, "top_k": int(top_k)}
        if bid_id.strip():
            params["bid_id"] = bid_id.strip()
        if doc_type:
            params["doc_type"] = doc_type
        try:
            result = api_get("/search", params)
        except Exception as exc:
            st.error(str(exc))
        else:
            for hit in result.get("hits") or []:
                st.markdown(f"**{hit.get('file_name')}** · page {hit.get('page_number')} · {hit.get('section_heading')}")
                st.caption(f"score {hit.get('score'):.4f} · {hit.get('bid_id')} · {hit.get('doc_type')}")
                st.write(hit.get("text"))
                st.divider()

elif page == "Ask":
    st.subheader("Ask the bids")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    bid_filter = st.text_input("Limit to bid folder (optional)")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            for citation in message.get("citations") or []:
                st.caption(f"{citation.get('file')} p.{citation.get('page')} {citation.get('section') or ''}")
    question = st.chat_input("Ask a question")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        payload = {"question": question}
        if bid_filter.strip():
            payload["bid_id"] = bid_filter.strip()
        try:
            result = api_post("/ask", payload)
            st.session_state.messages.append(
                {"role": "assistant", "content": result.get("answer"), "citations": result.get("citations") or []}
            )
        except Exception as exc:
            st.session_state.messages.append({"role": "assistant", "content": f"Request failed: {exc}"})
        st.rerun()

elif page == "Extract":
    st.subheader("Structured extraction")
    bid_id = st.text_input("Bid folder", value="Bid 1")
    if st.button("Run extraction", type="primary") and bid_id.strip():
        with st.spinner("Extracting fields from retrieved sections..."):
            try:
                record = api_post("/extract", {"bid_id": bid_id.strip()})
            except Exception as exc:
                st.error(str(exc))
                record = None
        if record:
            st.session_state["last_extract"] = record
    record = st.session_state.get("last_extract")
    if record and record.get("bid_id") == bid_id.strip():
        validation = record.get("validation") or {}
        st.write(
            f"Passed {validation.get('passed', 0)} · "
            f"Failed {validation.get('failed', 0)} · "
            f"Not found {validation.get('not_found', 0)}"
        )
        rows = []
        for name, field in (record.get("fields") or {}).items():
            sources = field.get("sources") or []
            source_text = "; ".join(f"{source.get('file')} p.{source.get('page')}" for source in sources)
            rows.append(
                {
                    "field": name,
                    "value": field.get("value"),
                    "confidence": field.get("confidence"),
                    "sources": source_text,
                    "notes": field.get("notes"),
                }
            )
        st.dataframe(rows, use_container_width=True)
        changes = record.get("addendum_changes") or []
        if changes:
            st.markdown("**Addendum changes**")
            st.dataframe(changes, use_container_width=True)
        st.download_button(
            "Download JSON",
            data=json.dumps(record, indent=2),
            file_name=f"{record.get('bid_id')}.json",
            mime="application/json",
        )

    st.divider()
    st.markdown("**Compare two stored extractions**")
    left_id = st.text_input("First bid", value="Bid 1", key="left_bid")
    right_id = st.text_input("Second bid", value="Bid 2", key="right_bid")
    if st.button("Load comparison"):
        columns = st.columns(2)
        for column, name in zip(columns, (left_id.strip(), right_id.strip())):
            with column:
                st.markdown(f"**{name}**")
                try:
                    loaded = api_get("/extractions", {"bid_id": name})
                except Exception as exc:
                    st.warning(str(exc))
                    continue
                for field_name, field in (loaded.get("fields") or {}).items():
                    st.markdown(f"*{field_name}*")
                    st.write(field.get("value") or field.get("notes"))
