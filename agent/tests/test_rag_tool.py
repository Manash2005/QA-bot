from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.tools.rag_tool import search_knowledge_base

def main():
    result = search_knowledge_base.invoke({
        "query": "What new school structure does the policy propose to replace the 10+2 structure, and which ages does it cover?",
        "top_k": 3
    })

    print(result)


if __name__ == "__main__":
    main()