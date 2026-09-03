# Upgraded MDT Prototype
New lead → CRM + Jira/Kanban + team knowledge + products + case studies + pricing + demos → specialized agents → orchestrator → customized sales pitch → PPT.
Run:
pip install -r requirements.txt
streamlit run app.py


## New: Enterprise Source Explorer
The sidebar now contains clickable enterprise sources. Each source opens its simulated records, connection status, record count, and the agents that consume it. This makes the agentic data-retrieval layer visible during the MDT demo.


## Better PPT output
The PPT generator uses fixed layouts, controlled text areas, cards, a solution architecture diagram, case-study callout, project evidence and commercial cards. It does not require a live Jira connection or an LLM.


## Expanded NexaAI data model
The prototype now represents a fictional industrial AI company, NexaAI Technologies, with expanded repositories for CRM, products, team knowledge, projects/Kanban, case studies, pricing, demos, industry knowledge, competitors, partners and sales history.

## Knowledge enrichment
The Sales Package tab can save an approved/generated opportunity into `data/sales_history.csv` and `data/crm.csv`. This demonstrates the knowledge-enrichment loop: generated sales intelligence -> saved opportunity -> future retrieval.


## Industry Context Gate
The app now starts by selecting a go-to-market industry context. That context becomes the retrieval boundary for the lead workflow and can be changed from the sidebar.
