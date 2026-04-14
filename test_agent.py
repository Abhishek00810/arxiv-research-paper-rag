import sys
sys.path.insert(0, 'src')
from agent import agent

result = agent.invoke({
    "query": "What is PagedAttention?",
    "original_query": "",
    "retrieved_chunks": [],
    "relevance_decision": "",
    "relevance_reason": "",
    "retry_count": 0,
    "final_answer": "",
    "sources": []
})

print("ANSWER:")
print(result["final_answer"])
print("\nSOURCES:")
for s in result["sources"]:
    print(s)
print("\nRETRIES:", result["retry_count"])
