import argparse
import uvicorn
import sys

def main():
    parser = argparse.ArgumentParser(description="LiveLearn (lile) Daemon CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start command
    start_parser = subparsers.add_parser("start", help="Start the lile daemon")
    start_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    start_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    if args.command == "start":
        print(f"Starting lile daemon on {args.host}:{args.port}...")
        uvicorn.run("lile.server:app", host=args.host, port=args.port, reload=False)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
