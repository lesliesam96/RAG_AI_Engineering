Here's a professional template you can use to add **How to Use** and **Demo Video** sections to your `README.md`:

---

###  How to Use

1. **Clone the repository**
   ```bash
   git clone https://github.com/lesliesam96/RAG_AI_Engineering.git
   cd RAG_AI_Engineering
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**  
   Create a `.env` file in the `app` directory and add your API keys:
   ```
   GENAI_KEY=your_genai_api_key
   GENAI_ENDPOINT=your_genai_endpoint
   ```

5. **Launch the app**
   ```bash
   streamlit run app/main.py
   ```

6. **Upload your file**  
   PDF, CSV, or TXT via the sidebar.

7. **Ask questions**  
   Use the chat box to ask questions about your document and get AI-powered insights.

---

### 🎥 Demo Video

[![Watch the Demo](demo_image.png)](https://link-to-your-demo-video.com)

>  *Click the image to watch the full demo*

---
