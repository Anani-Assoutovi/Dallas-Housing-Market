#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 17:01:12 2025

@author: Anani Assoutovi
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# We Load the CSV dataset stored in the same directory as the notebook

def load_data(file_path):
    # We read the CSV into a DataFrame
    df = pd.read_csv(file_path)  
    return df


# We clean and prepare the dataset

def clean_data(df):
    # We sandardize column names (remove spaces, use underscores)
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # We convert Payment_Date to datetime format
    if 'RUNDATE' in df.columns:
        df['RUNDATE'] = pd.to_datetime(df['RUNDATE'], errors='coerce')

    # We clean up Vendor_Name field: strip whitespace and standardize capitalization
    df['VENDOR'] = df['VENDOR'].str.strip().str.title()

    # We drop rows where crucial fields are missing
    df = df.dropna(subset=['VENDOR', 'CHKSUBTOT', 'RUNDATE'])

    # We ensure Amount column is numeric
    df['CHKSUBTOT'] = pd.to_numeric(df['CHKSUBTOT'], errors='coerce')
    
    # We drop rows where Amount conversion failed
    df = df.dropna(subset=['CHKSUBTOT'])  
    return df


# We print general statistics and missing value report

def generate_summary(df):
    print("Dataset Summary:")
    # We summarize all columns
    print(df.describe(include='all'))  
    print("\nMissing values per column:\n", df.isnull().sum())
    
    
# We Calculate total payment per vendor

def total_payments_by_vendor(df):
    vendor_totals = df.groupby('VENDOR')['CHKSUBTOT'].sum().sort_values(ascending=False)
    print("\nTop 10 Vendors by Total Payments:")
    print(vendor_totals.head(20))
    return vendor_totals


# We plot trend of total monthly payments
def monthly_trend(df):
    # Resample the Amount values by month
    monthly = df.set_index('RUNDATE').resample('ME')['CHKSUBTOT'].sum()

    # Plotting
    plt.figure(figsize=(12, 6))
    monthly.plot()
    plt.title('Monthly Vendor Payments')
    plt.ylabel('Total Amount')
    plt.xlabel('Date')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
# We identify outlier payments in the top 1% of amount

def detect_anomalies(df):
    # We identify the 99th percentile threshold
    threshold = df['CHKSUBTOT'].quantile(0.99)
    
    # We filter payments above the threshold
    outliers = df[df['CHKSUBTOT'] > threshold]
    print(f"\nPayments above 99th percentile (> {threshold:.2f}): {len(outliers)} rows")
    return outliers


# We count how many times each vendor has been paid

def payment_frequency(df):
    freq = df['VENDOR'].value_counts()
    print("\nVendors by Payment Frequency:")
    print(freq.head(20))
    return freq


# We plot heatmap of vendor spending by department (if Department column exists)

def department_vendor_heatmap(df):
    if 'DEPARTMENT' in df.columns:
        # We create pivot table with total payments by vendor and department
        pivot = df.pivot_table(index='VENDOR', columns='DEPARTMENT', values='CHKSUBTOT', aggfunc='sum', fill_value=0)

        # We plot heatmap for top 20 vendors only (for readability)
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot.head(20), cmap='YlGnBu', annot=True, fmt=".0f")
        plt.title("Top 20 Vendors by Department Spending")
        plt.tight_layout()
        plt.show()


# We complete workflow to run all steps in order
def run_analysis(file_path):
    # We load
    df = load_data(file_path)

    # We clean
    df = clean_data(df)

    # We summarize
    generate_summary(df)

    # We store vendor totals
    vendor_totals = total_payments_by_vendor(df)

    # We plot the monthly trend
    monthly_trend(df)

    # we store the anomaly detection
    anomalies = detect_anomalies(df)

    # We step through the frequency
    payment_frequency(df)

    # we store the Heatmap
    department_vendor_heatmap(df)

    # We export the results
    vendor_totals.to_csv("total_payments_by_vendor.csv")

    # We export the outliers
    anomalies.to_csv("anomalous_payments.csv")
    
    
    
    
# We execute our analysis on the dataset it
run_analysis("Vendor_Payments_2019_2025.csv")








