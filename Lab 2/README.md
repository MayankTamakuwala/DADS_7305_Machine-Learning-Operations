# Airflow Lab 2 - Iris Classification Pipeline

## Quick Start

The primary way to run this project is using the provided setup script:

```bash
# 1. Run the setup script to prepare the environment
./setup.sh

# 2. Start Airflow
docker compose up
```

This will automatically:
- Clean up any existing containers and volumes
- Create required Airflow directories
- Set up the proper user permissions
- Install necessary dependencies (pandas, scikit-learn)
- Start all Airflow services

Once running, access the Airflow UI at `http://localhost:8080` and look for the `Airflow_Mayank_Iris_DAG` in the DAGs list.

### ML Model

This script is designed for iris flower classification using Logistic Regression. It provides functionality to load the iris dataset, perform data preprocessing, build and save a Logistic Regression classification model, and evaluate the model performance. The implementation uses a file-based data pipeline approach instead of XCom serialization for improved performance and scalability.

#### Prerequisites

Before using this script, make sure you have the following libraries installed:

- pandas
- scikit-learn (sklearn)
- pickle
- json

#### Usage

You can use this script to perform iris classification as follows:

```python
# Load the data and save to file
data_file = load_data()

# Preprocess the data and save to file
preprocessed_file = data_preprocessing(data_file)

# Build and save the classification model
metrics = build_save_model(preprocessed_file, 'logistic_regression_model.sav')

# Load the saved model and evaluate it
result = load_model_elbow('logistic_regression_model.sav', metrics)
print(result)
```

#### Functions

1. **load_data():**
   - *Description:* Loads the iris dataset from sklearn, saves it to CSV file, and returns the file path.
   - *Usage:*
     ```python
     data_file = load_data()
     ```

2. **data_preprocessing(data_file_path)**
   - *Description:* Reads data from CSV file, performs data preprocessing, and saves preprocessed data to pickle file.
   - *Usage:*
     ```python
     preprocessed_file = data_preprocessing(data_file)
     ```

3. **build_save_model(preprocessed_file_path, filename)**
   - *Description:* Builds a Logistic Regression model, saves it to a file, and returns performance metrics.
   - *Usage:*
     ```python
     metrics = build_save_model(preprocessed_file, 'logistic_regression_model.sav')
     ```

4. **load_model_elbow(filename, metrics)**
   - *Description:* Loads a saved Logistic Regression model and evaluates it on new test data.
   - *Usage:*
     ```python
     result = load_model_elbow('logistic_regression_model.sav', metrics)
     ```

### Airflow Setup

Use Airflow to author workflows as directed acyclic graphs (DAGs) of tasks. The Airflow scheduler executes your tasks on an array of workers while following the specified dependencies.

References

-   Product - https://airflow.apache.org/
-   Documentation - https://airflow.apache.org/docs/
-   Github - https://github.com/apache/airflow

#### Installation

Prerequisites: You should allocate at least 4GB memory for the Docker Engine (ideally 8GB).

Local

-   Docker Desktop Running

Cloud

-   Linux VM
-   SSH Connection
-   Installed Docker Engine - [Install using the convenience script](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script)

#### Tutorial

1. Create a new directory

    ```bash
    mkdir -p ~/app
    cd ~/app
    ```

2. Running Airflow in Docker - [Refer](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html#running-airflow-in-docker)

    a. You can check if you have enough memory by running this command

    ```bash
    docker run --rm "debian:bullseye-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'
    ```

    b. Fetch [docker-compose.yaml](https://airflow.apache.org/docs/apache-airflow/2.5.1/docker-compose.yaml)

    ```bash
    curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.5.1/docker-compose.yaml'
    ```

    c. Setting the right Airflow user

    ```bash
    mkdir -p ./dags ./logs ./plugins ./working_data
    echo -e "AIRFLOW_UID=$(id -u)" > .env
    ```

    d. Update the following in docker-compose.yml

    ```bash
    # Donot load examples
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'

    # Additional python package
    _PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas scikit-learn }

    # Output dir
    - ${AIRFLOW_PROJ_DIR:-.}/working_data:/opt/airflow/working_data

    # Change default admin credentials
    _AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-airflow2}
    _AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-airflow2}
    ```

    e. Initialize the database

    ```bash
    docker compose up airflow-init
    ```

    f. Running Airflow

    ```bash
    docker compose up
    ```

    Wait until terminal outputs

    `app-airflow-webserver-1  | 127.0.0.1 - - [17/Feb/2023:09:34:29 +0000] "GET /health HTTP/1.1" 200 141 "-" "curl/7.74.0"`

    g. Enable port forwarding

    h. Visit `localhost:8080` login with credentials set on step `2.d`

3. Explore UI and add user `Security > List Users`

4. Create a python script [`dags/airflow.py`](dags/airflow.py)

    - PythonOperator
    - Task Dependencies
    - File-based data passing

    You can have n number of scripts inside dags dir

5. Stop docker containers

    ```bash
    docker compose down
    ```

### Airflow DAG Script

This Markdown file provides a detailed explanation of the Python script that defines an Airflow Directed Acyclic Graph (DAG) for an iris classification workflow.

#### Script Overview

The script defines an Airflow DAG named `Airflow_Lab` that consists of several tasks. Each task represents a specific operation in a machine learning classification workflow. The script imports necessary libraries, sets default arguments for the DAG, creates PythonOperators for each task, defines task dependencies, and provides command-line interaction with the DAG.

#### Importing Libraries

```python
# Import necessary libraries and modules
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow
```

The script starts by importing the required libraries and modules. Notable imports include the `DAG` and `PythonOperator` classes from the `airflow` package, datetime manipulation functions, and custom functions from the `src.lab` module.

#### Define default arguments for your DAG

```python
default_args = {
    'owner': "Mayank's Workflow",
    'start_date': datetime.now(),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}
```

Default arguments for the DAG are specified in a dictionary named default_args. These arguments include the DAG owner's name, the start date, the number of retries, and the retry delay in case of task failure.

#### Create a DAG instance named 'Airflow_Mayank_Iris_DAG' with the defined default arguments

```python
with DAG(
    'Airflow_Mayank_Iris_DAG',
    default_args=default_args,
    description='DAG for iris dataset classification using Logistic Regression',
    catchup=False,
) as dag:
```

Here, the DAG object is created with the name 'Airflow_Mayank_Iris_DAG' and the specified default arguments. The description provides a brief description of the DAG, and catchup is set to False to prevent backfilling of missed runs.

#### Task to load data, calls the 'load_data' Python function

```python
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
)
```

#### Task to perform data preprocessing, depends on 'load_data_task'

```python
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
)
```

The 'data_preprocessing_task' depends on the 'load_data_task' and calls the data_preprocessing function, which is provided with the output of the 'load_data_task'.

#### Task to build and save a Logistic Regression model, depends on 'data_preprocessing_task'

```python
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "logistic_regression_model.sav"],
)
```

The 'build_save_model_task' depends on the 'data_preprocessing_task' and calls the build_save_model function with specific arguments.

#### Task to load and evaluate the model using the 'load_model_elbow' function, depends on 'build_save_model_task'

```python
load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_elbow,
    op_args=["logistic_regression_model.sav", build_save_model_task.output],
)
```

The 'load_model_task' depends on the 'build_save_model_task' and calls the load_model_elbow function with specific arguments.

#### Set task dependencies

```python
load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task
```

Task dependencies are defined using the >> operator. In this case, the tasks are executed in sequence: 'load_data_task' -> 'data_preprocessing_task' -> 'build_save_model_task' -> 'load_model_task'.

#### If this script is run directly, allow command-line interaction with the DAG

```python
if __name__ == "__main__":
    dag.test()
```

- Lastly, the script allows for command-line interaction with the DAG. When the script is run directly, the dag.test() function is called, providing the ability to test the DAG from the command line.
- This script defines a comprehensive Airflow DAG for an iris classification workflow, with clear task dependencies and default arguments.

### Running an Apache Airflow DAG Pipeline in Docker

This guide provides detailed steps to set up and run an Apache Airflow Directed Acyclic Graph (DAG) pipeline within a Docker container using Docker Compose. The pipeline is named "Airflow_Lab."

#### Prerequisites

- Docker: Make sure Docker is installed and running on your system.

#### Step 1: Directory Structure

Ensure your project has the following directory structure:

```plaintext
your_airflow_project/
├── dags/
│   ├── airflow.py     # Your DAG script
│   ├── src/
│   │   └── lab.py     # Data processing and modeling functions
│   ├── data/          # Directory for data files
│   ├── model/         # Directory for model files
│   └── preprocessed/  # Directory for preprocessed data
├── docker-compose.yaml         # Docker Compose configuration
└── setup.sh                   # Setup script
```

#### Step 2: Docker Compose Configuration

Create a docker-compose.yaml file in the project root directory. This file defines the services and configurations for running Airflow in a Docker container.

#### Step 3: Start the Docker containers by running the following command

```plaintext
docker compose up
```

Wait until you see the log message indicating that the Airflow webserver is running:

```plaintext
app-airflow-webserver-1 | 127.0.0.1 - - [17/Feb/2023:09:34:29 +0000] "GET /health HTTP/1.1" 200 141 "-" "curl/7.74.0"
```

#### Step 4: Access Airflow Web Interface

- Open a web browser and navigate to http://localhost:8080.

- Log in with the credentials set in the .env file or use the default credentials (username: airflow, password: airflow).

- Once logged in, you'll be on the Airflow web interface.

#### Step 5: Trigger the DAG

- In the Airflow web interface, navigate to the "DAGs" page.

- You should see the "Airflow_Mayank_Iris_DAG" listed.

- To manually trigger the DAG, click on the "Trigger DAG" button or enable the DAG by toggling the switch to the "On" position.

- Monitor the progress of the DAG in the Airflow web interface. You can view logs, task status, and task execution details.

#### Step 6: Pipeline Outputs

- Once the DAG completes its execution, check the generated files in the respective directories:
  - `dags/data/` - Contains the iris dataset and metadata
  - `dags/preprocessed/` - Contains preprocessed data
  - `dags/model/` - Contains the trained model and metrics
  - `dags/model/evaluation/` - Contains evaluation results