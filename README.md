# Customer Segmentation

This project demonstrates unsupervised segmentation of customers based on annual income and spending score.  It uses K‑Means clustering to identify four distinct customer segments.

## Dataset

- `customers.csv` – synthetic dataset of 200 customers with fields `CustomerID`, `Gender`, `Age`, `AnnualIncome`, and `SpendingScore`.

## Analysis

The script `customer_segmentation.py` performs the following steps:

* Loads the dataset.
* Applies K‑Means clustering with four clusters using `AnnualIncome` and `SpendingScore`.
* Saves a cluster assignment file (`customers_with_clusters.csv`).
* Creates a scatter plot visualizing the clusters (`clusters.png`).

## Results

The resulting scatter plot shows clusters of customers differentiated by annual income and spending score.  The cluster assignments can be used for targeted marketing strategies.

## Usage

Run the script to reproduce the analysis:

```bash
python customer_segmentation.py
```