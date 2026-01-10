# Deployment Instructions (GitHub Method)

Since you prefer using GitHub, here is how to deploy your code to your Hetzner server.

## 1. Local Setup (Windows)
1.  **Create a Repository:** Go to [GitHub.com/new](https://github.com/new) and create a new public (or private) repository named `hackathon-2026`.
2.  **Push Code:**
    Open your terminal in this folder (`c:\Users\somya\OneDrive\Documents\Data hackathon`) and run:
    ```powershell
    git init
    git add .
    git commit -m "Initial commit"
    git branch -M main
    git remote add origin https://github.com/<YOUR_USERNAME>/hackathon-2026.git
    git push -u origin main
    ```
    *(Replace `<YOUR_USERNAME>` with your actual GitHub username)*

## 2. Server Setup (Hetzner)
1.  **SSH into your server:**
    ```bash
    ssh root@your_server_ip
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/<YOUR_USERNAME>/hackathon-2026.git
    cd hackathon-2026
    ```
    *(If you made it private, you will need to enter your username and a Personal Access Token).*

3.  **Install Dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```

## 3. Configure API Key
The `.env` file was excluded from Git for security. You must set the key on the server manually:

```bash
export GEMINI_API_KEY="AIzaSy..."
```
*(Replace `AIzaSy...` with your actual key)*

## 4. Run the Pipeline
```bash
python3 app.py
```

## 5. Check Results
```bash
head -n 5 results.csv
```
