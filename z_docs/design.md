# Design & UI Specifications

## 1. Frontend Framework
* **Primary Tool:** Streamlit (`streamlit`). Chosen for rapid data visualization and seamless Python integration.

## 2. Thematic Guidelines
* **Theme:** Dark Mode default (easier on the eyes for developer tools).
* **Color Palette:**
    * Background: `#0E1117` (Standard Streamlit Dark)
    * Accent / Primary Action: `#00FFAA` (Cyber Mint for successful hits)
    * Warning / Critique Loop: `#FFCC00` (Amber for state resets)
    * Error / Terminal: `#FF4B4B` (Red for failed loops)
* **Typography:**
    * Headings: `Inter` or `Roboto` (Sans-serif).
    * Metrics & Code Blocks: `Fira Code`, `JetBrains Mono`, or any monospaced font to emphasize the "developer tool" aesthetic.

## 3. Layout Structure
* **Sidebar:** * Configuration sliders (e.g., adjust the `k` constant for RRF).
    * Model selection (Groq vs Gemini toggle).
* **Main Canvas - Top:** Chat interface / Query input.
* **Main Canvas - Middle:** "Thinking" expander that visualizes the LangGraph state (showing if it had to loop and heal).
* **Main Canvas - Bottom (Dashboard):** 3-column layout displaying metrics: Latency (ms), Tokens used, and the RRF contribution split (**BM25 vs Vector vs Graph**).