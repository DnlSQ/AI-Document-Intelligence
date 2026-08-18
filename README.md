# AI Document Intelligence

> A local AI assistant designed to help technicians
> retrieve and understand technical documentation
> during equipment troubleshooting.

## Project Motivation

This project was inspired by my experience working as a
Module Technician at Intel.

During equipment troubleshooting, technicians often need
to consult datasheets, equipment manuals, procedures and
technical documentation to diagnose failures and restore
equipment as quickly as possible.

This led to the idea of developing a local AI assistant
capable of processing technical documentation and helping
technicians retrieve relevant information during repairs.

## Problem

Technical troubleshooting can require searching through
large amounts of documentation.

This can increase:

- Diagnosis time
- Equipment downtime
- Time spent searching for technical information
- Dependence on individual experience

## Project Goal

Build a local AI-powered document intelligence system that
can:

- Ingest technical documents
- Extract and clean text
- Index technical information
- Retrieve relevant information
- Generate contextual answers using a local LLM
- Provide references to the original documentation

## Real-World Background

During my experience at Intel, I contributed to improving
equipment availability from approximately 75% to more than
93% through:

- Process data analysis
- Identification of recurring failures
- Technical improvements
- Equipment troubleshooting
- Continuous improvement initiatives

I also developed Excel and Power BI tools to monitor
equipment performance, identify failure trends and support
data-driven decision making.

AI Document Intelligence is an evolution of this experience,
combining industrial troubleshooting, data analysis and
Artificial Intelligence.

## Architecture

Current architecture:

PDF
 ↓
Document Loader
 ↓
Text Extraction
 ↓
Text Cleaning
 ↓
[Future: Chunking]
 ↓
[Future: Embeddings]
 ↓
[Future: Vector Database]
 ↓
[Future: RAG]
 ↓
Local LLM

## Technologies

- Python
- Ollama
- Qwen 2.5 7B
- PyMuPDF
- Git
- PowerShell
- Local LLM inference

## Current Status

The project currently focuses on building a local AI
application using Python, Ollama and Qwen 2.5 7B.

The initial implementation includes local LLM inference
and conversational memory, with the architecture being
expanded toward document intelligence and RAG capabilities.

## Current Progress

- [x] Local LLM setup
- [x] Conversational memory
- [x] Project configuration
- [x] LLM abstraction
- [x] PDF ingestion
- [x] PDF text extraction
- [x] Text cleaning
- [ ] Text chunking
- [ ] Embeddings
- [ ] Vector database
- [ ] RAG pipeline
- [ ] Source attribution
- [ ] Technician-oriented interface

## Design Philosophy

The project is intentionally designed to run locally,
without requiring paid AI APIs.

This allows technical documentation to remain within
the local environment and demonstrates how LLM-based
applications can be developed with local models.

## Future Vision

The long-term goal is to transform the project into a
technical troubleshooting assistant capable of helping
technicians:

1. Identify relevant documentation
2. Search technical specifications
3. Investigate recurring failures
4. Retrieve troubleshooting procedures
5. Reduce diagnosis time
6. Improve equipment recovery time

## Author

Daniel Felipe Solano Quirós

B.Sc. Systems Engineering  
Electronics Technician

www.linkedin.com/in/daniel-felipe-solano-quiros

Costa Rica