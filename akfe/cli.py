import argparse
import json
import sys
from .engine import FusionEngine
from .api import app  # for type hinting only

def main():
    parser = argparse.ArgumentParser(
        prog="akfe",
        description="Adaptive Knowledge Fusion Engine command‑line interface"
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--api", action="store_true", help="Send query to the FastAPI server")
    parser.add_argument("--host", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    if args.api:
        import requests
        resp = requests.post(
            f"{args.host}/query",
            json={"query": args.query, "top_k": args.top_k}
        )
        resp.raise_for_status()
        results = resp.json()
    else:
        engine = FusionEngine()
        results = engine.query(args.query, top_k=args.top_k)

    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    sys.exit(main())
