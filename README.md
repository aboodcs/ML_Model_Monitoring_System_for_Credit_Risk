# 🔁 ML Workflow (Pipeline Stages)

## 1. 📥 Data Ingestion
Data is collected from different sources such as:
- CSV files
- Databases
- APIs

👉 **Output:** Raw dataset stored locally in the project structure

---

## 2. 🧹 Data Validation
This step ensures the dataset is correct, consistent, and usable for ML.

It includes:
- Checking schema (columns, data types)
- Detecting missing values
- Verifying data quality rules

👉 **Output:** Clean validated dataset or validation report

---

## 3. 🔄 Data Transformation
Data is prepared for machine learning models.

It includes:
- Encoding categorical features
- Scaling numerical features
- Feature engineering

👉 **Output:** Fully processed dataset ready for training

---

## 4. 🤖 Model Training
A machine learning model is built and trained.

It includes:
- Selecting the ML algorithm
- Training on processed data
- Saving trained model artifacts

👉 **Output:** Trained model file (`.pkl`, `.joblib`, etc.)

---

## 5. 📊 Model Evaluation
The trained model is evaluated to measure performance.

It includes:
- Calculating evaluation metrics (accuracy, RMSE, etc.)
- Comparing with baseline models
- Deciding whether the model is acceptable for deployment

---

# 🧱 Workflow Development Structure

When building an end-to-end ML pipeline, we follow a structured development order.  
The first three steps form the **foundation of every ML project** | **((((VERY IMPORTANT))))**.
## `config.yaml` , `schema.yaml` , `params.yaml`
---

## 1. ⚙️ Configuration Setup

Define all project settings and parameters:

- `config.yaml` → project paths, directories, pipeline configuration  
- `schema.yaml` → structure and validation rules of input data  
- `params.yaml` → model hyperparameters and tuning settings  

👉 **Purpose:** Centralize all configurations so the project is flexible and easy to manage.

---

## 2. 🧬 Entity Definitions

Define data classes (entities) for configurations and pipeline components.

👉 **Purpose:**
- Ensures type safety  
- Defines clear structure for each configuration section  
- Makes the pipeline more reliable and maintainable  

---

## 3. ⚙️ Configuration Manager (`src/config`)

Responsible for reading and managing configuration files.

It:
- Loads YAML files
- Converts them into Python objects
- Supplies them to components and pipelines

👉 **Purpose:** Provides a clean interface between configuration and code logic.

---

## 4. 🧩 Components Development

Each pipeline stage is implemented as an independent component:

- Data Ingestion Component  
- Data Validation Component  
- Data Transformation Component  
- Model Training Component  
- Model Evaluation Component  

👉 **Purpose:**
- Modular design  
- Reusable code  
- Easier debugging and testing  

---

## 5. 🔗 Pipeline Construction

All components are connected in a sequential workflow:

```text
Data Ingestion → Data Validation → Data Transformation → Model Training → Model Evaluation
```
