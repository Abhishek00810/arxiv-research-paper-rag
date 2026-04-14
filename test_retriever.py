import sys
sys.path.insert(0, 'src')
from retriever import retrieve

results = retrieve('What is PagedAttention?')
for r in results:
    print(r['title'], '|', r['section'])
    print(r['text'][:150])
    print('---')
