"""CLI for the RFP Intelligence Platform."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rfp_intel.config import get_settings
from rfp_intel.db.migrate import upgrade
from rfp_intel.logging_setup import setup_logging
from rfp_intel.service import ServiceError, ask_question, extract_bid, index_folder, search_bids


def main() -> None:
    parser = argparse.ArgumentParser(description="RFP Intelligence Platform")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start the API. Ingestion does not start until you run a pipeline step")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    index = sub.add_parser("index", help="Parse and index one bid folder now")
    index.add_argument("--bid", required=True, help="Path to a bid folder")

    search = sub.add_parser("search", help="Hybrid search")
    search.add_argument("query")
    search.add_argument("--bid", default=None)
    search.add_argument("--doc-type", default=None)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--mode", default="hybrid_rerank", choices=["vector", "hybrid", "hybrid_rerank"])

    ask = sub.add_parser("ask", help="Answer a question with citations")
    ask.add_argument("question")
    ask.add_argument("--bid", default=None)

    extract = sub.add_parser("extract", help="Extract the structured bid record")
    extract.add_argument("--bid", required=True, help="Bid folder name, for example 'Bid 1'")

    args = parser.parse_args()
    setup_logging(get_settings().log_level)

    if args.command == "serve":
        import uvicorn

        from api.app import app

        uvicorn.run(app, host=args.host, port=args.port)
        return

    upgrade()
    try:
        if args.command == "index":
            result = index_folder(args.bid, synchronous=True)
            print(json.dumps(result, indent=2))
        elif args.command == "search":
            hits = search_bids(
                args.query,
                bid_id=args.bid,
                doc_type=args.doc_type,
                top_k=args.top_k,
                mode=args.mode,
            )
            print(json.dumps([hit.model_dump() for hit in hits], indent=2))
        elif args.command == "ask":
            print(ask_question(args.question, args.bid).model_dump_json(indent=2))
        elif args.command == "extract":
            print(extract_bid(args.bid).model_dump_json(indent=2))
    except ServiceError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(exc.status_code) from exc


if __name__ == "__main__":
    main()
