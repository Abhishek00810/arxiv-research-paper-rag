#!/bin/bash

# Health check
echo "=== Health Check ==="
curl -s http://localhost:8000/health
echo ""

# Test query
echo ""
echo "=== Query: What is PagedAttention? ==="
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is PagedAttention?"}' | python3 -m json.tool
