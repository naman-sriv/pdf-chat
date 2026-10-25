# Retrieval evaluation

32 questions over `samples` (247 chunks).

| Method | Hit@1 | Hit@4 | MRR@10 | concept (hit@4) | exact-term (hit@4) | Latency |
| --- | --- | --- | --- | --- | --- | --- |
| bm25 | 44% | 59% | 0.54 | 12/24 | 7/8 | 0 ms |
| vector | 53% | 75% | 0.63 | 20/24 | 4/8 | 10 ms |
| hybrid | 50% | 78% | 0.65 | 18/24 | 7/8 | 18 ms |
| rerank | 62% | 84% | 0.74 | 19/24 | 8/8 | 342 ms |

### Misses (not in top 4)

- **bm25** missed 13: "How does the framework define risk?" (rank >10); "Does the framework tell organizations how much risk they should be willing to accept?" (rank 9); "Why is trying to get rid of all negative risk a bad idea?" (rank 6); "What problems come up when an AI system relies on data or software from outside vendors?" (rank 9); "Should AI risks be handled separately from an organization's other risks?" (rank 6); "What does validation mean in this framework?" (rank >10); "What is the difference between explaining how a model works and interpreting what its output means?" (rank 7); "What kinds of bias does NIST say organizations need to manage?" (rank 5); "What improvements might an organization see if using the framework is working?" (rank >10); "What are the main functions that make up the framework's core?" (rank 8); "Why aren't existing risk frameworks enough for AI systems?" (rank 7); "How much of the decision making can be left to the AI versus a person?" (rank 8); "What is MAP 5.1?" (rank 6)
- **vector** missed 8: "What improvements might an organization see if using the framework is working?" (rank 8); "When should AI systems be tested?" (rank >10); "Why aren't existing risk frameworks enough for AI systems?" (rank >10); "Is the framework meant to be readable by people who are not AI specialists?" (rank >10); "What does GOVERN 6.1 cover?" (rank 10); "What is MAP 5.1?" (rank >10); "What is MEASURE 2.11 about?" (rank 9); "What does MANAGE 2.4 say?" (rank >10)
- **hybrid** missed 7: "How does the framework define risk?" (rank 5); "What does validation mean in this framework?" (rank 5); "What improvements might an organization see if using the framework is working?" (rank >10); "When should AI systems be tested?" (rank 5); "Why aren't existing risk frameworks enough for AI systems?" (rank >10); "Is the framework meant to be readable by people who are not AI specialists?" (rank 6); "What does MANAGE 2.4 say?" (rank 7)
- **rerank** missed 5: "Does the framework tell organizations how much risk they should be willing to accept?" (rank 7); "What problems come up when an AI system relies on data or software from outside vendors?" (rank 6); "Who is the main intended audience of the framework?" (rank 8); "Which part of the core is about understanding the setting an AI system will operate in?" (rank >10); "How much of the decision making can be left to the AI versus a person?" (rank 5)
