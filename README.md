# Hofstadter Infinity Chat (HIC)

*Because your context window deserves better.*

## Overview

HIC is an intelligent chat application that solves the context window limitation of Large Language Models (LLMs). Using CHOFF (Cognitive Hoffman Compression Framework) markers for structured metadata, HIC maintains rich, long-running conversations without losing important context.

## Features

- 🧠 **Smart Context Management**: Store and retrieve conversation history with CHOFF metadata
- 🔍 **Intelligent Retrieval**: Find relevant parts of past conversations using content and CHOFF patterns
- 📝 **Dynamic Summarization**: Automatically condense older messages while preserving key information
- 🎯 **Focused Context**: Build minimal but sufficient prompts for each interaction
- 🔄 **Async-First**: Built with Trio for robust async/await support
- 🚀 **FastAPI Backend**: High-performance REST API for conversation management

## Design Philosophy

HIC is built on three core principles:

1. **Structured Memory**: Use CHOFF markers to add machine-readable metadata to conversations
2. **Minimal Context**: Only include what's relevant for the current interaction
3. **Robust Testing**: Property-based testing ensures reliability at scale

## Architecture

HIC consists of several key components:

- **Message Store**: Core storage system for conversations with CHOFF tag support
- **Retriever**: Intelligent system for finding relevant past messages
- **Summarizer**: Dynamic message condensation while preserving CHOFF metadata
- **LLM Orchestrator**: Smart prompt assembly and LLM interaction management
- **FastAPI Backend**: RESTful API exposing HIC's capabilities

## Development

HIC leverages modern Python tooling:
- Python 3.13+
- FastAPI for the web layer
- Trio for async operations
- Hypothesis for property-based testing

Built with ❤️ and recursive self-reference