# AIDS Lab Practicals

Python programs for the 10 experiments in the AIDS Lab Manual.

## Setup
1. Install Python 3.x.
2. Clone/download this private repository.
3. Open a terminal in the project folder and run:
   `python -m pip install -r requirements.txt`
4. Run a program, for example: `python Practical-01_Data-Preprocessing/prog1.py`

Some datasets are bundled with scikit-learn (Wine, Iris). Titanic and Flights are loaded by Seaborn and require internet access the first time. CIFAR-10 is downloaded by TensorFlow/Keras on first run.

## Practicals
| No. | Program | Dataset |
|---|---|---|
| 01 | Data Cleaning and Preprocessing | employee_raw_data.csv (included) |
| 02 | Exploratory Data Analysis | Titanic (Seaborn) |
| 03 | Linear Regression | Study hours vs marks (in code) |
| 04 | Logistic Regression | Study hours vs marks (in code) |
| 05 | Decision Tree and Random Forest | Wine (scikit-learn) |
| 06 | K-Means Clustering | Iris (scikit-learn) |
| 07 | PCA | Iris (scikit-learn) |
| 08 | Simple Neural Network from Scratch | XOR (in code) |
| 09 | CNN Image Classification | CIFAR-10 (TensorFlow/Keras) |
| 10 | Time Series Forecasting | Flights (Seaborn) |

The programs follow the experiment aims and procedures in the supplied AIDS Lab Manual. Dataset loaders may need internet access on first use.
