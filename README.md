### A framework to disentangle health-related information from speaker embeddings

Speaker embeddings are used to recognize the user identity, which may contain different para-linguistic attributes, such as prosody, emotion, phonation, articulation, etc. We hypothesize that:

1. some of the attributes are related to health condition;
2. by properly disentangling the health-related information from speaker embeddings, a health-preserving voice anonymization model can then be developed;
3. the disentangled health embeddings could be speaker-antagonist and more generalizable.


We investigate the following questions:

#### Question-1: Do speaker embeddings carry health information?

- Task: To predict if an individual has a respiratory symptom (e.g., cough, headache, fever etc.)
- Dataset: Cambridge COVID-19 Sound Database
- Pipeline: Pre-trained speaker embeddings (x-vector and/or ECAPA-TDNN embeddings) + FC
- Training Strategy: Conventional BCE loss / Contrastive loss

#### Question-2: How to disentangle health info from speaker embeddings?

* Approach: Parallel multi-task learning
* Diagram: [PLACEHOLDER]

#### Question-3: The performance of resultant speaker embeddings and health embeddings?
