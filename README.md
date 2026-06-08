# PRODIGY_ML_01 - Retail Customer Segmentation using K-Means Clustering

## Overview

This project applies the **K-Means Clustering** algorithm to segment retail customers based on their purchasing behavior and demographic characteristics. Customer segmentation helps businesses identify different groups of customers and develop targeted marketing strategies, improve customer retention, and increase sales.

## Objective

The objective of this project is to group customers into distinct clusters based on features such as:

* Age
* Annual Income
* Total Spending
* Order Count
* Average Basket Value
* Number of Product Categories Purchased

By identifying customer segments, businesses can better understand customer behavior and make data-driven decisions.

---

## Dataset Features

| Feature       | Description                            |
| ------------- | -------------------------------------- |
| CustomerID    | Unique identifier for each customer    |
| Age           | Customer age                           |
| AnnualIncome  | Customer annual income                 |
| TotalSpend    | Total amount spent by the customer     |
| OrderCount    | Total number of orders placed          |
| AvgBasket     | Average basket value per order         |
| NumCategories | Number of product categories purchased |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## Methodology

1. Load and preprocess customer data
2. Select relevant features
3. Standardize features using StandardScaler
4. Determine the optimal number of clusters using the Elbow Method
5. Apply K-Means Clustering
6. Visualize customer segments using PCA
7. Analyze cluster characteristics and customer behavior

---

## Project Structure

```text
PRODIGY_ML_01/
│
├── customer_segmentation.py
├── retail_customer_segmentation_dataset.csv
├── requirements.txt
├── README.md
└── screenshots/
    ├── elbow_method.png
    └── cluster_visualization.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Dorjit-Nameirakpam/PRODIGY_ML_01.git
```

Navigate to the project directory:

```bash
cd PRODIGY_ML_01
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python customer_segmentation.py
```

## Results

The K-Means model successfully segments customers into meaningful groups based on spending habits and purchasing behavior. These segments can represent:

* High-value customers
* Premium shoppers
* Frequent buyers
* Budget-conscious customers
* Occasional shoppers

Such segmentation can be used for personalized marketing campaigns, customer retention strategies, and business growth initiatives.

---

## Learning Outcomes

Through this project, the following concepts were explored:

* Unsupervised Machine Learning
* K-Means Clustering
* Data Preprocessing
* Feature Scaling
* Principal Component Analysis (PCA)
* Data Visualization
* Customer Analytics

---

## Author

**Dorjit Nameirakpam**

GitHub: https://github.com/Dorjit-Nameirakpam

---

## License

This project is developed as part of the Prodigy InfoTech Machine Learning Internship Program.
