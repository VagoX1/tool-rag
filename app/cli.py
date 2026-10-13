from __future__ import annotations

import argparse
import json

import uvicorn

from .agent import AgentResult, RAGAgent
from .config import Settings, require_api_key
from .langchain_rag import LangChainRAG


def print_result(result: AgentResult) -> None:
    print(result.answer)
    print("\nMetadata:")
    print(
        json.dumps(
            {
                "session_id": result.session_id,
                "citations": result.citations,
                "web_sources": result.web_sources,
                "citation_warnings": result.citation_warnings,
                "tool_calls": result.tool_calls,
                "usage": result.usage,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-enabled local RAG starter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Incrementally rebuild the local index")
    subparsers.add_parser("status", help="Show indexed documents and chunk counts")
    subparsers.add_parser("sessions", help="List locally stored chat sessions")
    clear_parser = subparsers.add_parser("clear-session", help="Delete one chat session")
    clear_parser.add_argument("session_id")

    search_parser = subparsers.add_parser("search", help="Search without generation")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=4)
    search_parser.add_argument("--source", action="append", dest="sources")

    ask_parser = subparsers.add_parser("ask", help="Ask one question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--session")
    ask_parser.add_argument("--source", action="append", dest="sources")
    ask_parser.add_argument("--no-web", action="store_true")

    langchain_parser = subparsers.add_parser(
        "langchain-ask", help="Ask only the indexed documents through LangChain"
    )
    langchain_parser.add_argument("question")
    langchain_parser.add_argument("--source", action="append", dest="sources")
    langchain_parser.add_argument("--top-k", type=int)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive conversation")
    chat_parser.add_argument("--session")
    chat_parser.add_argument("--source", action="append", dest="sources")
    chat_parser.add_argument("--no-web", action="store_true")

    serve_parser = subparsers.add_parser("serve", help="Run the HTTP API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.command == "serve":
        uvicorn.run("app.api:app", host=args.host, port=args.port, reload=False)
        return

    require_api_key()
    agent = RAGAgent(Settings())
    if args.command == "ingest":
        print(json.dumps(agent.store.ingest(), indent=2))
        return
    if args.command == "status":
        print(json.dumps(agent.store.status(), indent=2))
        return
    if args.command == "sessions":
        print(json.dumps(agent.sessions.sessions(), indent=2))
        return
    if args.command == "clear-session":
        deleted = agent.sessions.clear(args.session_id)
        print(json.dumps({"session_id": args.session_id, "deleted_messages": deleted}, indent=2))
        return
    if args.command == "search":
        results = agent.store.search(args.query, args.top_k, args.sources)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    if args.command == "ask":
        print_result(
            agent.ask(
                args.question,
                args.session,
                args.sources,
                allow_web=not args.no_web,
            )
        )
        return
    if args.command == "langchain-ask":
        result = LangChainRAG(agent.store, agent.settings).ask(
            args.question,
            sources=args.sources,
            top_k=args.top_k,
        )
        print(result.answer)
        print("\nMetadata:")
        print(
            json.dumps(
                {
                    "framework": "langchain",
                    "citations": result.citations,
                    "citation_warnings": result.citation_warnings,
                    "retrieved_chunks": result.retrieved_chunks,
                },
                indent=2,
            )
        )
        return

    session_id = args.session or agent.sessions.new_id()
    print(f"Session: {session_id}")
    print("Type 'exit' or 'quit' to finish.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = agent.ask(
            question, session_id, args.sources, allow_web=not args.no_web
        )
        print(f"\nAssistant: {result.answer}\n")
        if result.web_sources:
            print("Web sources:")
            for source in result.web_sources:
                print(f"- {source['title']}: {source['url']}")
            print()
        if result.citation_warnings:
            print("Citation warnings:", "; ".join(result.citation_warnings), "\n")


if __name__ == "__main__":
    main()
