# Gmail-Assistant


Gmail AI Assistant – Project Overview

This project is a fully AI-powered Gmail automation system that connects Gmail, Ollama, and Milvus to create an intelligent email assistant. It securely links to your Gmail account and automatically reads incoming emails. Each email is then processed and transformed into numerical vector embeddings using a local Ollama model, such as Llama or Mistral.

These embeddings are stored and managed in Milvus, a high-performance vector database that helps the system understand the meaning and relationships between emails. When a new message arrives, the assistant retrieves relevant context from Milvus and generates a smart, context-aware reply using Ollama’s language model.

The generated response is then saved as a draft in Gmail, allowing users to review and send it manually. The system’s modular structure makes it easy to modify or extend individual components, such as the email reader, embedding generator, database connector, or reply module.

Overall, this project demonstrates how local AI models and vector databases can work together to create a privacy-focused, context-aware email assistant capable of understanding, managing, and responding to real conversations intelligently.