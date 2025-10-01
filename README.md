# 🧪 Acne Project

Acne Project is a Django-based web application for **skin health classification** (Healthy / Low / High level) using region-based image analysis.  

---

## 🚀 Run with Docker (Recommended)

### Prerequisites
- Install [Docker](https://docs.docker.com/get-docker/)  
- Install [Docker Compose v2](https://docs.docker.com/compose/install/)  
- Git

### Setup
Clone the repository:
```bash
git clone https://github.com/spyros0202/Ance_project.git
cd Ance_project
```

### Build and Run
From the project root:

```bash
docker compose up --build
```

The app will be available at:  
👉 http://127.0.0.1:8000

Stop containers:
```bash
docker compose down
```

If you want a **full cleanup** (remove containers, images, volumes, and networks):
```bash
docker system prune -a --volumes -f
```

---

## 🐍 Run Locally with Conda/venv

### Prerequisites
- [Anaconda / Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Python 3.11+
- Git

### Setup
Clone the repository:
```bash
git clone https://github.com/spyros0202/Ance_project.git
cd Ance_project
```

Create a new conda environment:
```bash
conda create -n acne_project python=3.11
conda activate acne_project
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run database migrations:
```bash
python manage.py migrate
```

Start the development server:
```bash
python manage.py runserver
```

The app will be available at:  
👉 http://127.0.0.1:8000

---

## 📦 Standalone `.exe` (PyInstaller)

If you don’t want to install Python or dependencies, a standalone Windows `.exe` is available.  
It was built using **PyInstaller** and bundles the GUI app + ML models.  

📩 Contact me if you would like the executable version.  

⚠️ The `.exe` only runs on Windows. For Linux/macOS, use Docker or Conda setup.

---

## 🖼 Example Images

The repository also includes an **`images/`** folder with sample images.  
You can use these as test inputs to quickly try out the web app or the standalone GUI.

---

## 🛠 Development Notes
- **Static files** are stored in `/static` and collected into `/staticfiles` during Docker builds.  
- **Database**: Default is SQLite (`db.sqlite3`).  
- **Gunicorn** is used in Docker for production.  
