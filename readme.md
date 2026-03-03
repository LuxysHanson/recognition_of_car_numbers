# Install env

```bash
conda create -n recognition_numbers python=3.13.5 -y
conda activate recognition_numbers
pip install -r requirements.txt
```

# Train

```bash
python train_model.py 
```

### Run the DOCKER

```bash
docker-compose up --build
```

* Test check: `GET http://localhost:8501` → `{"status":"ok"}`
