# 🚀 Deployment Guide for Food Delivery Predictor

This guide will walk you through deploying your Streamlit application to the web using **Render.com**. Render is a cloud platform that offers a free tier perfect for hosting Streamlit apps.

---

## 📋 Prerequisites

Before we begin, ensure you have:
1.  **GitHub Account**: You will need this to host your code. [Sign up here](https://github.com/join).
2.  **Render Account**: You can sign up using your GitHub account. [Sign up here](https://render.com/).
3.  **Git Installed**: Ensure Git is installed on your local machine.

---

## Phase 1: Prepare Your Repository

We need to put your code onto GitHub so Render can access it.

### 1. Initialize Git (Locally)
Open your terminal in the project folder (`e:\INTTRVU_HACKATHON`) and run:

```bash
# Initialize a new git repository
git init

# Add all files to the repository
git add .

# Commit the files
git commit -m "Initial commit of Delivery Prediction App"
```

### 2. Create a Repository on GitHub
1.  Go to [GitHub.com](https://github.com) and sign in.
2.  Click the **+** icon in the top-right and select **New repository**.
3.  Name it `food-delivery-predictor`.
4.  Make it **Public**.
5.  Click **Create repository**.

### 3. Push Code to GitHub
Copy the commands shown on the GitHub "Quick setup" page (under "...or push an existing repository from the command line") and run them in your terminal. They will look like this:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/food-delivery-predictor.git
git push -u origin main
```
*(Replace `YOUR_USERNAME` with your actual GitHub username)*

---

## Phase 2: Deploy on Render

Now we connect Render to your GitHub repository.

1.  **Dashboard**: Go to your [Render Dashboard](https://dashboard.render.com/).
2.  **New Web Service**: Click the **New +** button and select **Web Service**.
3.  **Connect GitHub**: You may need to authorize Render to access your GitHub account. Once done, select your new `food-delivery-predictor` repository from the list.
4.  **Configure Settings**:
    *   **Name**: `delivery-predictor-app` (or any unique name).
    *   **Region**: Choose the one closest to you (e.g., *Singapore* or *Oregon*).
    *   **Branch**: `main`.
    *   **Root Directory**: Leave blank (or `.`).
    *   **Runtime**: **Python 3**.
    *   **Build Command**: 
        ```bash
        pip install -r requirements.txt
        ```
    *   **Start Command**: 
        ```bash
        streamlit run app.py
        ```
5.  **Free Tier**: Scroll down and select the **Free** instance type.
6.  **Deploy**: Click **Create Web Service**.

---

## ⏳ What Happens Next?

1.  Render will start building your app. You will see scrolling logs in the dashboard.
2.  It will install all libraries from `requirements.txt`.
3.  It will run the `streamlit run app.py` command.
4.  **Success**: Once you see "Your service is live", click the URL provided (e.g., `https://delivery-predictor-app.onrender.com`) to view your live app!

---

## 🔧 Troubleshooting

- **"Model not found" error**: 
    - Ensure `best_delivery_time_model.pkl` was included in your Git commit. Check your GitHub repo to ensure the file is there. NB: GitHub has a 100MB file limit. If your model is larger, you may need to use [Git LFS](https://git-lfs.github.com/).
- **"Module not found" error**:
    - Check your `requirements.txt` file. Every library imported in `app.py` must be listed there.
