# AskDB – AI-Powered Text-to-SQL Assistant  
*“Talk to your database like you talk to people.”* 💬📊  

AskDB is an **AI-powered text-to-SQL assistant** that lets users interact with databases using plain English. Built with 🧠 **LLMs (Gemini / OpenAI), 🖥️ Flask, and 🎨 TailwindCSS**, it automatically converts natural language into SQL queries, executes them, and returns insights with **beautiful analytics and visualizations**.

---

## 🧠 Problem Statement  
Writing SQL queries can be challenging for non-technical users.  
> ❓ *“Can we ask our database questions in plain English?”*  
> ✅ *Yes – with AskDB.*

AskDB eliminates the need to know SQL syntax. Simply type your question (e.g., *“Show me the total sales by region this year”*), and the app automatically generates, executes, and explains the SQL output for you.

---

## 🌐 Interface Preview  

### Dashboard (Light Mode)
*(Add screenshots here once ready)*

### Dark Mode Analytics
(Add screenshot here)

---

## 🏗️ System Architecture
```text
User Query ➡️ AskDB Interface ➡️ LLM Engine (Gemini / GPT)
                    ⬇️                      ⬆️
          SQL Generation 🔁 SQL Execution on Database
                    ⬇️
          Query Results ➡️ Analytics ➡️ Visual Insights
```

---

## 🚀 Key Features

- 💬 Natural Language Querying – Ask data-related questions in plain English.
- 🤖 LLM-Powered SQL Generation – Converts text to SQL using Gemini / GPT.
- 🗄️ Database Connectivity – Connects to your local or cloud SQL databases.
- 📊 Advanced Query Analytics – Displays visual charts, trends, and summaries.
- 🧠 Evaluation Metrics – BLEU, ROUGE, METEOR, and RAGAS for SQL accuracy.
- 🌗 Modern UI Themes – Classic dark and light themes built with TailwindCSS.
- 🔍 Error Insights – Highlights invalid queries and suggests corrections.
- 📈 Interactive Visualizations – Generates charts for SQL results automatically.

---

## Project Structure

| File / Folder           | Description                                       |
| ----------------------- | ------------------------------------------------- |
| `app.py`                | Flask backend for text-to-SQL conversion          |
| `templates/`            | Frontend HTML templates with Jinja2               |
| `static/`               | CSS, JS, and image assets (AskDB logo, icons)     |
| `evaluation_metrics.py` | BLEU, ROUGE, METEOR, and RAGAS evaluation logic   |
| `config.yaml`           | LLM API keys and database config (ignored in git) |
| `database/`             | Example SQLite or MySQL sample database           |
| `requirements.txt`      | Python dependencies                               |
| `README.md`             | Project documentation (this file)                 |

---

## 🔧 Technologies Used

- 🧠 LLM Engine: Gemini / GPT-4 / GPT-5
- 🖥️ Backend: Flask (Python)
- 💾 Database: SQLite / MySQL (customizable)
- 🎨 Frontend: TailwindCSS, Jinja2 Templates
- 📊 Analytics: Matplotlib, Pandas, Plotly
- 🧮 Evaluation: BLEU, ROUGE, METEOR, RAGAS'

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Manthan2110/AskDB-AI-Assistant.git
cd AskDB-AI-Assistant
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # For Mac/Linux
venv\Scripts\activate         # For Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure API Key
Edit config.yaml and add your Gemini or OpenAI key:
```bash
GEMINI_API_KEY: "your_api_key_here"
```

### 5️⃣ Run Flask App
```bash
python app_enhanced.py
```

--- 

## 🧮 Sample Evaluation Output
```bash
{
  "User_Input": "How many products did Unit Ltd purchase in 2021?",
  "Generated_SQL": "SELECT COUNT(*) FROM purchases WHERE company='Unit Ltd' AND YEAR(date)=2021;",
  "Reference_SQL": "SELECT COUNT(*) FROM purchases WHERE company='Unit Ltd' AND YEAR(date)=2021;",
  "BLEU": 0.84,
  "ROUGE-1": 1.0,
  "ROUGE-L": 1.0,
  "METEOR": 0.87
}
```

---

## 📈 How It Works
- The user types a question in plain English
- LLM converts it into an optimized SQL query
- The SQL query runs on the connected database
- Results are displayed in table and chart form
- Evaluation metrics assess SQL generation accuracy

---

## 🎯 Future Enhancements

| Feature                        | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| 🧠 **SQL Semantic Similarity** | Add cosine similarity scoring for deeper accuracy |
| 🗃️ **Multi-DB Support**       | PostgreSQL, Snowflake, and MongoDB connectors     |
| 💬 **Conversational Memory**   | Maintain context across multiple user queries     |
| 📊 **Dynamic Visualization**   | Auto-select best chart type for results           |
| 🧩 **User Authentication**     | Personalized query history and dashboards         |
| ⚙️ **API Access**              | Enable external API integration for developers    |

---

## 👨‍💻 Author
Made with 🧠 and ❤️ by Manthan Jadav
- [🌐LinkedIn](https://www.linkedin.com/in/manthanjadav/)
- [💻GitHub](https://github.com/Manthan2110)
- 📧 manthanjadav746@gmail.com

---

## 📜 License
This project is licensed under the MIT License.
Ask freely. Query smartly. Build intelligently. 🚀


