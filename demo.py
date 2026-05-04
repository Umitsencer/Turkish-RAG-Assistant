"""Demo: index a document and ask questions.

Usage:
    python demo.py --docs docs/ornek.txt --query "Turkiye'nin baskenti neresidir?"
"""
import argparse
from src.rag_pipeline import TurkishRAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Turkish RAG Demo")
    parser.add_argument("--docs", nargs="+", required=True, help="Belge yollari (.txt veya .pdf)")
    parser.add_argument("--query", required=True, help="Sorulacak soru")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    print("\n Turkish RAG Assistant - Baslatiliyor...")
    rag = TurkishRAGPipeline()
    rag.build_vectorstore(args.docs)
    rag.build_qa_chain(top_k=args.top_k)
    result = rag.query(args.query)

    print(f"\nSoru : {result['question']}")
    print(f"Yanit: {result['answer']}")
    print("\nKaynak Parcalar:")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] {src['source']} -> {src['content'][:120]}...")


if __name__ == "__main__":
    main()
