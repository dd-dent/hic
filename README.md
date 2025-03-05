# Hofstadter Infinity Chat (HIC)

*Because your context window deserves better.*

## Overview

HIC is an intelligent chat application that solves the context window limitation of Large Language Models (LLMs) through structured metadata and smart context management. Using CHOFF (Cognitive Hoffman Compression Framework) markers, HIC maintains rich, long-running conversations without losing important context.

```mermaid
graph TD
    A[User Message] --> B[Message Store]
    B --> C{Context Manager}
    C -->|Pattern Match| D[Retriever]
    C -->|Compress Old| E[Summarizer]
    D --> F[Context Assembly]
    E --> F
    F --> G[LLM Interface]
    G --> H[Response]
    H --> B
```

## Key Features

- 🧠 **Smart Context Management**
  - CHOFF metadata for structured state tracking
  - Pattern-based relevance scoring
  - Dynamic context assembly

```mermaid
graph LR
    A[Raw Message] --> B[CHOFF Parser]
    B --> C{State Tracker}
    C -->|Patterns| D[Pattern Store]
    C -->|States| E[State Store]
    D --> F[Context Builder]
    E --> F
```

- 🔍 **Intelligent Retrieval**
  - CHOFF-aware message search
  - Pattern evolution tracking
  - State transition validation

- 📝 **Dynamic Summarization**
  - Metadata-preserving compression
  - Pattern-aware condensation
  - State transition tracking

```mermaid
sequenceDiagram
    participant U as User
    participant M as Message Store
    participant R as Retriever
    participant S as Summarizer
    participant L as LLM

    U->>M: New Message
    M->>R: Find Relevant Context
    R->>S: Check Old Messages
    S-->>R: Compressed History
    R-->>M: Relevant Context
    M->>L: Assembled Prompt
    L-->>U: Response
```

## Architecture

HIC uses a modular, event-driven architecture:

```mermaid
graph TD
    subgraph Storage Layer
        MS[Message Store]
        ES[Event Store]
    end
    
    subgraph Core Services
        R[Retriever]
        S[Summarizer]
        CM[Context Manager]
    end
    
    subgraph Interface Layer
        API[FastAPI Backend]
        WS[WebSocket Server]
    end
    
    API --> CM
    WS --> CM
    CM --> R
    CM --> S
    R --> MS
    S --> MS
    CM --> ES
```

### Components

- **Message Store**: SQLite-based persistence with CHOFF tag support
- **Event Store**: Tracks conversation events and state transitions
- **Retriever**: CHOFF-aware message search and relevance scoring
- **Summarizer**: Intelligent message condensation preserving metadata
- **Context Manager**: Dynamic context assembly and state tracking

## Development

HIC leverages modern Python tooling:
- Python 3.13+
- FastAPI for the web layer
- Trio for async operations
- SQLite for persistence
- Hypothesis for property-based testing

## Getting Started

1. Clone the repository
```bash
git clone https://github.com/yourusername/hic.git
cd hic
```

2. Set up Python environment
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Run the tests
```bash
pytest
```

4. Start the server
```bash
uvicorn hic.api:app --reload
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Built with ❤️ and recursive self-reference

---

*"In the face of context collapse, structure emerges."*