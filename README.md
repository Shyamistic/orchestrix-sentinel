# 🛡️ Orchestrix Sentinel
### The Zero-Trust, Self-Healing Neural Orchestration Layer for IBM Watsonx

![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-Production%20Ready-green)
![Tech](https://img.shields.io/badge/IBM-Watsonx-blue)

---

## 🚨 The Enterprise AI Problem
While Agentic AI promises automation, Enterprises are stalled by **"Black Box Anxiety"**:
1.  **Compliance Risk:** An autonomous agent might unknowingly violate GDPR, HIPAA, or internal finance policies (e.g., transferring funds to a sanctioned entity).
2.  **Operational Fragility:** If a critical API (SAP, Salesforce) throws a `503` error, standard agents crash, breaking the entire workflow.
3.  **Tool Stagnation:** Agents are limited to hard-coded tools and cannot adapt to novel data tasks.
4.  **The "Why?" Gap:** Executives lack visibility into *why* an agent made a high-stakes decision.

## 💡 The Solution: Orchestrix Sentinel
**Orchestrix Sentinel** is not a chatbot. It is a **Governance Mesh** that sits between the user and IBM Watsonx Orchestrate. It acts as a **Guardian**, **Healer**, and **Toolmaker**, ensuring agents operate safely, reliably, and transparently.

---

## 🏗️ System Architecture

Sentinel operates as a **Deterministic State Machine** wrapped in a **Neural Interface**.

```mermaid
graph TD
    User[User / Voice Command] -->|HTTPS/WSS| FE[Frontend: Neural Dashboard]
    FE -->|REST API| BE[Backend: Orchestrix Sentinel Core]
    
    subgraph "Layer 1: The Neural Core"
        BE -->|Log Event| Ledger[Immutable Trust Ledger (SQL + SHA256)]
        BE -->|Vectorize Prompt| Embed[SentenceTransformers]
        Embed -->|Semantic Search| Chroma[ChromaDB: Policy Vector Store]
        BE -->|Forecast Load| DT[Digital Twin Simulator]
    end
    
    subgraph "Layer 2: Governance Mesh"
        Chroma -->|Risk Score| Mesh[Multi-Agent Consensus]
        Mesh -->|Vote| Planner[Planner Agent]
        Mesh -->|Vote| Safety[Safety Agent]
        Mesh -->|Vote| Comp[Compliance Agent]
    end
    
    subgraph "Layer 3: Execution & Healing"
        Mesh -->|Approved| ADK[ADK Bridge]
        ADK -->|Execute| WX[IBM Watsonx Orchestrate]
        WX -->|Action| EXT[External APIs (SAP/Slack)]
        
        EXT --x|Error (503)| Healer[Self-Healing Kernel]
        Healer -->|Pause & Backoff| WX
    end
    
    subgraph "Layer 4: JIT Evolution"
        BE -->|Missing Tool| Gen[Genesis Engine]
        Gen -->|Prompt| Granite[IBM Granite-13b]
        Granite -->|Python Code| Linter[Security AST Analyzer]
        Linter -->|Hot-Load| WX
    end

####

Data Flow Pipeline
1) Ingestion: User intent is captured via Voice/
Text.

2) Vector Guardrail: The intent is embedded (all-MiniLM-L6-v2) and checked against uploaded PDF/YAML policies in ChromaDB.

3) Governance Vote: A "Supreme Court" of agents (Safety, Planner, Compliance) votes on the action.

4) Execution: If approved, the request is routed to IBM Watsonx Orchestrate via the ADK Bridge.

5) Resilience: If an API fails, the Self-Healing Kernel detects the error pattern and auto-retries.

6) Audit: Every step is cryptographically hashed and stored in the Immutable Trust Ledger.

####

Key Features (The "Acqui-hire" Suite)
1. Federated Guardian Engine (RAG)
Intercepts high-risk prompts before execution. Unlike simple keyword filters, it uses Vector Search to understand the context of a policy violation (e.g., blocking "move cash offshore" based on anti-money laundering docs).

2. Self-Healing Kernel
Simulates enterprise chaos. If a workflow step fails (e.g., API Timeout), Sentinel:

a) Detects the specific exception.

b) Pauses the state machine.

c) Retries with exponential backoff.

d) Visualizes the healing process (Red Node → Blue Node).

3. "Why Not?" Explainability Engine
Executives don't just need to know what happened; they need to know what didn't happen.

a) Feature: Generates a causal graph of rejected paths.

b) Example: "I did not auto-approve this invoice because the amount > $10k and Policy FIN-01 requires manual review."

4. JIT (Just-in-Time) Skill Genesis
Solves agent stagnation. If the user asks for a tool that doesn't exist (e.g., "Calculate crypto taxes"), Sentinel:

a) Prompts IBM Granite to write the code.

b) Runs a Security Linter (AST) to block dangerous imports (os, sys).

c) Hot-loads the skill into the runtime instantly.

5. Digital Twin & AIOps Monitor
A live SVG Topology Map of the organization. It allows managers to simulate stress scenarios (e.g., "Hiring surge +20%") and predicts SLA breaches using an ARIMA-style forecasting model.

📊 Business Value (ROI)
Our dashboard doesn't just show logs; it shows money saved.

a) Risk Avoided: Calculated based on the regulatory fines associated with blocked policy violations (e.g., GDPR = 4% of revenue).

b) SLA Uptime: Calculated based on minutes saved by the Self-Healing module preventing manual IT triage.

🛠️ Tech Stack
Frontend: Modern Vanilla JS, Tailwind CSS (Glassmorphism), HTML5 Canvas (Neural Mesh), Chart.js.

Backend: Python FastAPI, Uvicorn.

AI/ML: PyTorch, SentenceTransformers, ChromaDB.

Persistence: SQLite (SQLAlchemy), NetworkX (Graph Theory).

Infrastructure: Docker-ready, 12-Factor App compliant.

🚀 Getting Started
Prerequisites
Python 3.9+

IBM Cloud Account (for Watsonx API keys)

Installation
Clone the Repository

Bash

git clone [https://github.com/Shyamistic/orchestrix-sentinel.git](https://github.com/YOUR_USERNAME/orchestrix-sentinel.git)
cd orchestrix-sentinel
Install Dependencies

Bash

pip install -r requirements.txt
Configure Environment Create a .env file in the backend folder:

Ini, TOML

IBM_WATSONX_API_KEY=your_api_key_here
IBM_ORCHESTRATE_ENDPOINT=[https://api.us-south.orchestrate.ibm.com](https://api.us-south.orchestrate.ibm.com)
Run the System

Bash

uvicorn backend.main:app --reload
Launch Mission Control Open frontend/index.html in your browser.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

Built for the IBM Agentic AI Hackathon 2025.