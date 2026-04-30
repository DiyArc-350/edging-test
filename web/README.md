# Web HTML Generator

This directory contains a bash script (`makeweb.sh`) that automatically generates 100 random HTML gallery pages using a set of provided images.

## Prerequisites

- A Unix-like environment (Linux, macOS, or WSL on Windows) with Bash installed.
- (Optional) **Docker and Docker Compose** if you want to view the files as a website.

## Step 1: Generate the Files

1. Open your terminal and navigate to this directory (`web`).
2. Make sure the script is executable (if not already):
   ```bash
   chmod +x makeweb.sh
   ```
3. Run the script:
   ```bash
   ./makeweb.sh
   ```
   *Alternatively, you can run it with `bash makeweb.sh`.*

**Output:**
Once the script finishes, it will generate a new folder named `html_files`. 
Inside this folder, you will find `index1.html` through `index100.html` and a `media` folder containing the images.

## Step 2: Serve as a Website (Using Docker)

If you want to view the generated galleries in your browser like a real website, you can use the provided Docker Compose setup which uses Nginx.

1. Ensure the `html_files` directory has been generated from Step 1.
2. Build and run the Docker container in detached mode:
   ```bash
   docker-compose up -d
   ```
3. Open your browser and navigate to: `http://localhost:8082/index1.html`
   *(You can change the number from 1 to 100 to view the different galleries).*

## How to Stop the Website

To stop serving the files, run:
```bash
docker-compose down
```
