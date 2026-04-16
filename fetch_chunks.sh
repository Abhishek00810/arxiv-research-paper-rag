#!/bin/bash

OUTPUT_FILE="evals/chunks_output.txt"
> $OUTPUT_FILE  # clear file

questions=(
    "What is PagedAttention and what problem does it solve?"
    "How does Firecracker achieve strong isolation between VMs?"
    "What is the core idea behind LoRA fine-tuning?"
    "What problem does Flash Attention solve compared to standard attention?"
    "What is retrieval-augmented generation RAG?"
    "How does Flash Attention reduce memory complexity?"
    "What are the tradeoffs between LoRA and full fine-tuning?"
    "How does Firecracker design differ from traditional containers?"
    "What is the role of the KV cache in LLM inference?"
    "How does vLLM use PagedAttention to improve throughput?"
    "memory management Firecracker PagedAttention similar problems"
    "relationship original attention mechanism Flash Attention optimization"
    "LoRA parameter efficiency scaling transformer"
    "Firecracker microVM vs Linux containers isolation performance"
    "RAG complement transformer limitations attention"
)

for i in "${!questions[@]}"; do
    echo "=== Question $((i+1)): ${questions[$i]} ===" | tee -a $OUTPUT_FILE

    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"${questions[$i]}\"}")

    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" == "200" ]; then
        echo "$BODY" | python3 -m json.tool >> $OUTPUT_FILE
    else
        echo "FAILED (HTTP $HTTP_CODE): $BODY" >> $OUTPUT_FILE
    fi

    echo "" >> $OUTPUT_FILE
    echo "Done question $((i+1))/15 — HTTP $HTTP_CODE"

    # wait between questions to avoid hitting token rate limits
    sleep 15
done

echo "All done. Results in $OUTPUT_FILE"
