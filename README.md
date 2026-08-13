# Live Market Sentiment API

> A high-performance, real-time market sentiment analysis backend API built with **FastAPI**, **Hugging Face Transformers**, and deployed on a **24/7 AWS EC2 cloud infrastructure**.

---

## 🚀 Features

* **Real-Time Sentiment Pipeline:** Evaluates live text data and market headlines using state-of-the-art transformer models.
* **Production-Ready Architecture:** Built using **FastAPI** for asynchronous handling, fast execution, and automatic interactive documentation.
* **Cloud-Native Deployment:** Hosted securely on an AWS EC2 Ubuntu instance, managed persistently via **systemd**, and accessible over a public IP endpoint.
* **Developer Friendly:** Includes interactive Swagger UI (`/docs`) and ReDoc endpoints out of the box.

---

## 📂 Project Structure

```text
sentiment_analyzer/
│
├── main.py                  # FastAPI application entry point & route handlers
├── model.py                 # Core sentiment analysis & transformer logic
├── requirement.txt          # Project dependencies (CPU-optimized PyTorch & tools)
├── sentiment_tech_stocks.csv# Dataset source for market tracking
└── .gitignore               # Excludes virtual environments and sensitive keys
```
🛠️ Tech Stack
Backend Framework: FastAPI, Uvicorn

Machine Learning / NLP: PyTorch (CPU build), Hugging Face Transformers

Data Processing: Pandas, NumPy

Infrastructure & DevOps: AWS EC2 (Ubuntu Linux), Systemd Service Manager, Git & GitHub
