"""
================================================================================
PROJECT: Single-File Interactive EDA Dashboard
Teacher / Classroom Edition: Exploratory Data Analysis & Visualization
Dataset: Students Performance in Exams (Kaggle)
Technologies: Python, Pandas, NumPy, Matplotlib, Seaborn, Flask, HTML, CSS, JS
================================================================================
"""

import os
import io
import base64
import json
import urllib.parse
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np

# Use non-interactive backend for headless web rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Set global Seaborn and Matplotlib aesthetics
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14
})

# Configurable CSV path
CSV_PATH = os.environ.get("CSV_PATH", "StudentsPerformance.csv")


# ==============================================================================
# 1. LOAD DATASET
# ==============================================================================
def load_dataset(csv_path="StudentsPerformance.csv"):
    """
    Loads the student performance dataset from a CSV file.
    Returns (DataFrame, error_message).
    """
    if not os.path.exists(csv_path):
        # Look in current working directory and script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, os.path.basename(csv_path))
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            return None, f"Dataset file '{csv_path}' was not found in the working directory."
    
    try:
        df = pd.read_csv(csv_path)
        return df, None
    except Exception as e:
        return None, f"Error reading CSV file: {str(e)}"


# ==============================================================================
# 2. CLEAN DATASET
# ==============================================================================
def clean_dataset(df):
    """
    Cleans and prepares the dataset:
    - Strips leading/trailing whitespace from column headers and string values
    - Validates numeric columns and coerces invalid numbers
    """
    if df is None:
        return None
    
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()
    
    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            cleaned[col] = cleaned[col].astype(str).str.strip()
            
    # Ensure numeric columns are properly typed
    numeric_targets = ["math score", "reading score", "writing score"]
    for col in numeric_targets:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            
    return cleaned


# ==============================================================================
# 3. GET DATASET SUMMARY
# ==============================================================================
def get_dataset_summary(df):
    """
    Computes a comprehensive statistical and structural summary of the dataset.
    """
    if df is None:
        return {}
    
    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)
    
    summary = {
        "total_students": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "avg_math": float(round(df["math score"].mean(), 2)) if "math score" in df else 0.0,
        "avg_reading": float(round(df["reading score"].mean(), 2)) if "reading score" in df else 0.0,
        "avg_writing": float(round(df["writing score"].mean(), 2)) if "writing score" in df else 0.0,
        "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
        "duplicate_count": int(df.duplicated().sum()),
        "preview": df.head(8).to_dict(orient="records"),
        "stats": df.describe().round(2).to_dict()
    }
    return summary


# ==============================================================================
# 4. GET NUMERIC COLUMNS
# ==============================================================================
def get_numeric_columns(df):
    """Returns a list of numeric column names from the DataFrame."""
    if df is None:
        return []
    return df.select_dtypes(include=[np.number]).columns.tolist()


# ==============================================================================
# 5. GET CATEGORICAL COLUMNS
# ==============================================================================
def get_categorical_columns(df):
    """Returns a list of categorical / object column names from the DataFrame."""
    if df is None:
        return []
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


# ==============================================================================
# HELPER: FIGURE TO BASE64
# ==============================================================================
def fig_to_base64(fig):
    """Converts a Matplotlib figure to a base64 encoded PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


# ==============================================================================
# 6. CREATE HISTOGRAM
# ==============================================================================
def create_histogram(df, column):
    """
    Creates a detailed distribution histogram with KDE curve and mean/median markers.
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, f"Column '{column}' is not numeric.", ha="center", va="center")
        return fig_to_base64(fig)
    
    series = df[column].dropna()
    mean_val = series.mean()
    median_val = series.median()
    
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(series, kde=True, bins=20, color="#3b82f6", edgecolor="#1e40af", alpha=0.65, ax=ax)
    
    ax.axvline(mean_val, color="#ef4444", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color="#10b981", linestyle="-.", linewidth=2, label=f"Median: {median_val:.2f}")
    
    ax.set_title(f"Histogram: Distribution of {column.title()}", fontweight="bold", pad=12)
    ax.set_xlabel(column.title(), fontweight="semibold")
    ax.set_ylabel("Student Count (Frequency)", fontweight="semibold")
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    
    return fig_to_base64(fig)


# ==============================================================================
# 7. CREATE BAR CHART
# ==============================================================================
def create_bar_chart(df, x_column, y_column=None):
    """
    Creates a bar chart comparing group metrics (mean score) or frequency.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    if y_column and pd.api.types.is_numeric_dtype(df[y_column]):
        # Grouped average comparison
        grouped = df.groupby(x_column)[y_column].mean().reset_index()
        palette = sns.color_palette("Blues_r", n_colors=len(grouped))
        bars = ax.bar(grouped[x_column], grouped[y_column], color=palette, edgecolor="#1e3a8a", width=0.55)
        
        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
            
        ax.set_title(f"Bar Chart: Average {y_column.title()} by {x_column.title()}", fontweight="bold", pad=12)
        ax.set_ylabel(f"Average {y_column.title()}", fontweight="semibold")
        ax.set_ylim(0, max(100, grouped[y_column].max() * 1.15))
    else:
        # Category frequency count
        counts = df[x_column].value_counts()
        bars = ax.bar(counts.index, counts.values, color="#6366f1", edgecolor="#3730a3", width=0.55)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{int(height)}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(f"Bar Chart: Count by {x_column.title()}", fontweight="bold", pad=12)
        ax.set_ylabel("Count", fontweight="semibold")
        
    ax.set_xlabel(x_column.title(), fontweight="semibold")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    
    return fig_to_base64(fig)


# ==============================================================================
# 8. CREATE LINE CHART
# ==============================================================================
def create_line_chart(df, x_column=None, y_column=None):
    """
    Creates an educational line chart for meaningful ordered comparisons.
    Pedagogy: Line charts require an ordered sequence. We illustrate two cases:
    1. Average subject scores across logically ordered parental education levels
    2. Sorted score progression percentiles
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    edu_order = [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
    
    # Case 1: If x_column is parental level of education, use the natural education progression
    if x_column == "parental level of education" or (not x_column and not y_column):
        filtered_edu = [e for e in edu_order if e in df["parental level of education"].values]
        grouped = df.groupby("parental level of education")[["math score", "reading score", "writing score"]].mean().reindex(filtered_edu)
        
        ax.plot(grouped.index, grouped["math score"], marker="o", linewidth=2.5, label="Math Score", color="#3b82f6")
        ax.plot(grouped.index, grouped["reading score"], marker="s", linewidth=2.5, label="Reading Score", color="#10b981")
        ax.plot(grouped.index, grouped["writing score"], marker="^", linewidth=2.5, label="Writing Score", color="#f59e0b")
        
        ax.set_title("Line Chart: Mean Exam Scores across Parental Education Levels", fontweight="bold", pad=12)
        ax.set_xlabel("Parental Level of Education (Ordered)", fontweight="semibold")
        ax.set_ylabel("Average Score (0-100)", fontweight="semibold")
        ax.set_ylim(40, 90)
        ax.legend(loc="upper left", frameon=True)
        plt.xticks(rotation=25, ha="right")
    elif y_column and pd.api.types.is_numeric_dtype(df[y_column]):
        # Case 2: Sorted Student Score Percentiles (Cumulative Progression)
        sorted_scores = df[y_column].sort_values().reset_index(drop=True)
        percentiles = (sorted_scores.index / (len(sorted_scores) - 1)) * 100
        
        ax.plot(percentiles, sorted_scores, color="#6366f1", linewidth=2.5, label=f"Sorted {y_column.title()}")
        ax.axhline(sorted_scores.mean(), color="#ef4444", linestyle="--", label=f"Mean ({sorted_scores.mean():.1f})")
        ax.axhline(sorted_scores.median(), color="#10b981", linestyle="-.", label=f"Median ({sorted_scores.median():.1f})")
        
        ax.set_title(f"Line Chart: Cumulative Percentile Curve of {y_column.title()}", fontweight="bold", pad=12)
        ax.set_xlabel("Student Percentile Rank (%)", fontweight="semibold")
        ax.set_ylabel(f"{y_column.title()} (0-100)", fontweight="semibold")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper left", frameon=True)
    else:
        # Fallback to sorted index progression
        ax.plot(df.index[:100], df["math score"].iloc[:100], marker=".", color="#3b82f6", alpha=0.7)
        ax.set_title("Line Chart: Math Scores (First 100 Students Sample)", fontweight="bold")
        ax.set_xlabel("Student Index")
        ax.set_ylabel("Math Score")
        
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 9. CREATE BOX PLOT
# ==============================================================================
def create_box_plot(df, column, category_column=None):
    """
    Creates a box plot showing median, quartiles, IQR, and outliers.
    Supports univariate box plot or grouped by category.
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, f"Column '{column}' is not numeric.", ha="center", va="center")
        return fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    if category_column and category_column in df.columns:
        sns.boxplot(data=df, x=category_column, y=column, hue=category_column, legend=False,
                    palette="Set2", ax=ax, width=0.5,
                    flierprops={"marker": "o", "color": "#ef4444", "alpha": 0.6})
        ax.set_title(f"Box Plot: {column.title()} grouped by {category_column.title()}", fontweight="bold", pad=12)
        ax.set_xlabel(category_column.title(), fontweight="semibold")
        plt.xticks(rotation=20, ha="right")
    else:
        sns.boxplot(y=df[column], color="#93c5fd", width=0.35, ax=ax,
                    flierprops={"marker": "o", "markerfacecolor": "#ef4444", "alpha": 0.7})
        ax.set_title(f"Box Plot: Five-Number Summary of {column.title()}", fontweight="bold", pad=12)
        
        # Annotate Q1, Median, Q3
        q1 = df[column].quantile(0.25)
        med = df[column].median()
        q3 = df[column].quantile(0.75)
        ax.text(0.25, q1, f"Q1: {q1:.1f}", va="center", color="#1e3a8a", fontweight="bold")
        ax.text(0.25, med, f"Median: {med:.1f}", va="center", color="#b91c1c", fontweight="bold")
        ax.text(0.25, q3, f"Q3: {q3:.1f}", va="center", color="#1e3a8a", fontweight="bold")
        
    ax.set_ylabel(column.title(), fontweight="semibold")
    ax.set_ylim(-5, 105)
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 10. CREATE SCATTER PLOT
# ==============================================================================
def create_scatter_plot(df, x_column, y_column, hue_column=None):
    """
    Creates a scatter plot showing correlation, trendline, and optional category hue.
    """
    if not (pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column])):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Scatter plots require both X and Y to be numeric columns.", ha="center", va="center")
        return fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calculate Pearson correlation coefficient
    corr = df[x_column].corr(df[y_column])
    
    if hue_column and hue_column in df.columns:
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column, palette="tab10",
                        alpha=0.7, s=45, ax=ax)
        ax.legend(title=hue_column.title(), loc="upper left", frameon=True)
    else:
        sns.scatterplot(data=df, x=x_column, y=y_column, color="#2563eb", alpha=0.6, s=45, ax=ax)
        
    # Add regression trend line
    sns.regplot(data=df, x=x_column, y=y_column, scatter=False, ax=ax,
                color="#dc2626", line_kws={"linestyle": "--", "linewidth": 2})
    
    corr_desc = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
    ax.set_title(f"Scatter Plot: {x_column.title()} vs {y_column.title()} (r = {corr:.2f}, {corr_desc} Positive)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(x_column.title(), fontweight="semibold")
    ax.set_ylabel(y_column.title(), fontweight="semibold")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 11. CREATE HEATMAP
# ==============================================================================
def create_heatmap(df):
    """
    Creates a correlation matrix heatmap of all numeric exam scores.
    """
    num_df = df.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()
    
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", vmin=0.5, vmax=1.0, fmt=".3f",
                square=True, linewidths=1.5, cbar_kws={"label": "Pearson Correlation (r)"}, ax=ax)
    
    ax.set_title("Correlation Heatmap: Exam Subject Scores", fontweight="bold", pad=15)
    ax.set_xticklabels([c.title() for c in corr_matrix.columns], rotation=15, ha="right", fontweight="semibold")
    ax.set_yticklabels([c.title() for c in corr_matrix.columns], rotation=0, fontweight="semibold")
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 12. CREATE COUNT PLOT
# ==============================================================================
def create_count_plot(df, column):
    """
    Creates a categorical frequency count plot with numerical annotations on bars.
    """
    if column not in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, f"Column '{column}' not found.", ha="center", va="center")
        return fig_to_base64(fig)
        
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]
    
    fig, ax = plt.subplots(figsize=(8, 4.8))
    palette = sns.color_palette("crest", n_colors=len(counts))
    bars = ax.bar(counts[column], counts["count"], color=palette, edgecolor="#064e3b", width=0.55)
    
    total = len(df)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.annotate(f"{int(h)} ({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax.set_title(f"Count Plot: Frequency of {column.title()}", fontweight="bold", pad=12)
    ax.set_xlabel(column.title(), fontweight="semibold")
    ax.set_ylabel("Number of Students", fontweight="semibold")
    ax.set_ylim(0, counts["count"].max() * 1.15)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 13. CREATE PIE CHART
# ==============================================================================
def create_pie_chart(df, column):
    """
    Creates a clean pie chart showing percentage proportions of a categorical column.
    """
    if column not in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, f"Column '{column}' not found.", ha="center", va="center")
        return fig_to_base64(fig)
        
    counts = df[column].value_counts()
    
    # Colors and explode configuration
    colors = sns.color_palette("pastel", len(counts))
    explode = [0.03] * len(counts)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=explode,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    
    for at in autotexts:
        at.set_color("#1f2937")
        at.set_fontsize(10)
        at.set_fontweight("bold")
        
    ax.set_title(f"Pie Chart: Proportions of {column.title()}", fontweight="bold", pad=12)
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 14. CREATE PAIRPLOT
# ==============================================================================
def create_pairplot(df, hue_column="gender"):
    """
    Creates a Seaborn pairplot showing pairwise relationships across all scores.
    """
    num_cols = get_numeric_columns(df)
    cols_to_plot = list(num_cols)
    
    if hue_column and hue_column in df.columns:
        cols_to_plot.append(hue_column)
        g = sns.pairplot(df[cols_to_plot], hue=hue_column, palette="tab10",
                         diag_kind="kde", plot_kws={"alpha": 0.6, "s": 30})
    else:
        g = sns.pairplot(df[cols_to_plot], diag_kind="kde",
                         plot_kws={"color": "#3b82f6", "alpha": 0.6, "s": 30})
        
    g.fig.subplots_adjust(top=0.94)
    g.fig.suptitle(f"Pair Plot: Multivariate Score Comparison (Hue: {hue_column.title()})",
                   fontsize=14, fontweight="bold")
    
    buf = io.BytesIO()
    g.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(g.fig)
    return f"data:image/png;base64,{img_b64}"


# ==============================================================================
# 15. GET CHART EXPLANATION
# ==============================================================================
def get_chart_explanation(chart_type, x_col=None, y_col=None, hue_col=None):
    """
    Returns classroom pedagogical explanations for any generated graph.
    """
    explanations = {
        "histogram": {
            "title": f"Histogram Distribution: {x_col.title() if x_col else 'Numerical Variable'}",
            "purpose": "Shows the shape, frequency distribution, and spread of a single continuous numeric variable.",
            "when_to_use": "Use a histogram during univariate analysis to check whether scores are normally distributed, skewed left or right, or contain outliers.",
            "what_to_observe": "Notice where the peak lies (mode), whether the Mean (dashed red) and Median (green) are close together (symmetric) or pulled apart by low scores (skewed).",
            "data_types": "1 Continuous Numerical column (e.g. math score, reading score)."
        },
        "box_plot": {
            "title": f"Box Plot Summary: {x_col.title() if x_col else 'Metric'}",
            "purpose": "Visualizes the five-number summary: Minimum, Q1 (25th percentile), Median (50th), Q3 (75th), and Maximum, along with individual outlier points.",
            "when_to_use": "Use box plots to spot extreme outliers and compare score spreads (Interquartile Range - IQR) across demographic categories.",
            "what_to_observe": "Check the line inside the box (median), the box height (IQR spread), and individual points extending beyond the whiskers (outliers who scored significantly lower).",
            "data_types": "1 Numerical column (univariate) OR 1 Categorical + 1 Numerical (bivariate)."
        },
        "bar_chart": {
            "title": f"Bar Chart: Comparison by {x_col.title() if x_col else 'Category'}",
            "purpose": "Compares an aggregated summary metric (like average score) across discrete categorical groups.",
            "when_to_use": "Use bar charts when comparing distinct categories (e.g., parental education levels, lunch types) to determine which group averages higher.",
            "what_to_observe": "Compare bar heights to quickly identify top-performing and lower-performing groups. Notice the difference in mean performance between groups.",
            "data_types": "1 Categorical X column + 1 Aggregated Numerical Y column."
        },
        "line_chart": {
            "title": "Line Chart: Ordered Score Trends",
            "purpose": "Shows continuous trends, progression curves, or patterns across an ordered sequence or ranking.",
            "when_to_use": "Use line charts ONLY when the X-axis has a natural, meaningful order (e.g. increasing education level or sorted student percentiles).",
            "what_to_observe": "Observe the slope: does parental education correlate with steadily increasing subject scores? Notice how reading and writing track each other closely.",
            "data_types": "1 Ordered Categorical or Ranked Numerical X + 1 or more Numerical Y metrics."
        },
        "scatter_plot": {
            "title": f"Scatter Plot: {x_col.title() if x_col else 'X'} vs {y_col.title() if y_col else 'Y'}",
            "purpose": "Examines the bivariate relationship, direction, and strength of correlation between two continuous variables.",
            "when_to_use": "Use scatter plots to determine if two exam scores rise together (positive correlation), move inversely, or have no relationship.",
            "what_to_observe": "Look at point clustering along the red dashed trendline. A tight diagonal cluster indicates a high Pearson correlation coefficient (r > 0.8).",
            "data_types": "2 Continuous Numerical variables (optional 3rd Categorical Hue)."
        },
        "heatmap": {
            "title": "Correlation Matrix Heatmap",
            "purpose": "Displays pairwise Pearson correlation coefficients (r) between all numerical variables simultaneously using color intensity.",
            "when_to_use": "Use during multivariate exploration to quickly identify which features share strong statistical associations and which are independent.",
            "what_to_observe": "Values range from -1.0 to +1.0. Notice the very high correlation between Reading and Writing (~0.95), while Math has a slightly lower correlation with verbal skills.",
            "data_types": "All Numerical variables in the dataset."
        },
        "count_plot": {
            "title": f"Count Plot: Distribution of {x_col.title() if x_col else 'Category'}",
            "purpose": "Displays the exact frequency (count and percentage) of observations in each category.",
            "when_to_use": "Use during univariate categorical analysis to understand demographic representation and ensure data is balanced.",
            "what_to_observe": "Check for class imbalances (e.g. how many students completed test prep vs none, or proportions of standard vs free/reduced lunch).",
            "data_types": "1 Categorical variable."
        },
        "pie_chart": {
            "title": f"Pie Chart: Proportions of {x_col.title() if x_col else 'Category'}",
            "purpose": "Shows the relative proportions and percentage breakdown of parts making up a whole (100%).",
            "when_to_use": "Best used for categorical variables with few unique values (2 to 5 categories) to illustrate share and composition.",
            "what_to_observe": "Observe the relative wedge sizes. Notice whether categories are split nearly evenly (e.g. gender 52% vs 48%) or significantly skewed.",
            "data_types": "1 Categorical variable with few unique classes."
        },
        "pairplot": {
            "title": f"Pair Plot: Multi-Variable Interaction (Hue: {hue_col.title() if hue_col else 'None'})",
            "purpose": "A grid of scatter plots and distribution curves showing every pairwise combination of numeric features simultaneously.",
            "when_to_use": "Use during multivariate analysis to detect clustering, separate groups by demographic categories, and identify multidimensional outliers.",
            "what_to_observe": "Look at the diagonal density plots to see group separation, and off-diagonal scatter plots to check if group dots separate or overlap.",
            "data_types": "Multiple Numerical columns + 1 Categorical column for Hue."
        }
    }
    
    return explanations.get(chart_type, {
        "title": "Exploratory Data Analysis Visualization",
        "purpose": "Visualizes patterns and relationships in the data.",
        "when_to_use": "Choose the visualization best suited for the data types being analyzed.",
        "what_to_observe": "Examine distribution, central tendency, spread, and outliers.",
        "data_types": "Numerical or Categorical."
    })


# ==============================================================================
# 16. GENERATE MATPLOTLIB CODE
# ==============================================================================
def generate_matplotlib_code(chart_type, x_col=None, y_col=None, hue_col=None):
    """
    Generates beginner-friendly Matplotlib code for the specified chart.
    """
    x = x_col or "math score"
    y = y_col or "reading score"
    
    codes = {
        "histogram": f"""import matplotlib.pyplot as plt

# 1. Create canvas figure
plt.figure(figsize=(8, 5))

# 2. Plot histogram with explicit bins and borders
plt.hist(df['{x}'], bins=20, color='royalblue', edgecolor='black', alpha=0.7)

# 3. Add statistical reference lines
mean_val = df['{x}'].mean()
median_val = df['{x}'].median()
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {{mean_val:.2f}}')
plt.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {{median_val:.2f}}')

# 4. Set informative labels and title
plt.title('Distribution of {x.title()} (Matplotlib)', fontsize=14, fontweight='bold')
plt.xlabel('{x.title()}', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()""",

        "bar_chart": f"""import matplotlib.pyplot as plt

# 1. Manually calculate group averages first
avg_scores = df.groupby('{x}')['{y}'].mean()

# 2. Create figure and bar plot
plt.figure(figsize=(8, 5))
bars = plt.bar(avg_scores.index, avg_scores.values, color='#4C72B0', edgecolor='#1A365D', width=0.55)

# 3. Annotate values on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{{height:.1f}}',
             ha='center', va='bottom', fontweight='bold')

plt.title('Average {y.title()} by {x.title()} (Matplotlib)', fontsize=14, fontweight='bold')
plt.xlabel('{x.title()}', fontsize=12)
plt.ylabel('Mean {y.title()}', fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.show()""",

        "box_plot": f"""import matplotlib.pyplot as plt

# 1. Prepare data groups manually
groups = [group['{x}'].values for _, group in df.groupby('{y_col or "gender"}')]
labels = df['{y_col or "gender"}'].unique()

# 2. Create box plot
plt.figure(figsize=(7, 5))
plt.boxplot(groups, labels=labels, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red', linewidth=2))

plt.title('Box Plot of {x.title()} (Matplotlib)', fontsize=14, fontweight='bold')
plt.ylabel('{x.title()}', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()""",

        "scatter_plot": f"""import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 6))

# 1. Scatter points
plt.scatter(df['{x}'], df['{y}'], color='royalblue', alpha=0.6, edgecolors='none')

# 2. Add trendline with numpy polyfit
m, b = np.polyfit(df['{x}'], df['{y}'], 1)
plt.plot(df['{x}'], m * df['{x}'] + b, color='red', linestyle='--', linewidth=2, label='Trendline')

plt.title('Scatter Plot: {x.title()} vs {y.title()} (Matplotlib)', fontsize=14, fontweight='bold')
plt.xlabel('{x.title()}', fontsize=12)
plt.ylabel('{y.title()}', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()""",

        "heatmap": """import matplotlib.pyplot as plt
import numpy as np

# 1. Calculate correlation matrix
numeric_df = df[['math score', 'reading score', 'writing score']]
corr = numeric_df.corr().values
cols = numeric_df.columns

# 2. Display matrix using matshow
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(corr, cmap='YlGnBu', vmin=0.5, vmax=1.0)
fig.colorbar(cax)

# 3. Add tick labels and cell text annotations
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=20, ha='left')
ax.set_yticklabels(cols)

for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', color='black', fontweight='bold')

plt.title('Correlation Matrix (Matplotlib)', pad=25, fontweight='bold')
plt.tight_layout()
plt.show()""",

        "count_plot": f"""import matplotlib.pyplot as plt

counts = df['{x}'].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(counts.index, counts.values, color='mediumseagreen', edgecolor='black')
plt.title('Count of Students by {x.title()} (Matplotlib)', fontsize=14, fontweight='bold')
plt.xlabel('{x.title()}', fontsize=12)
plt.ylabel('Student Count', fontsize=12)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.show()""",

        "pie_chart": f"""import matplotlib.pyplot as plt

counts = df['{x}'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
        startangle=140, colors=['#66b3ff', '#99ff99', '#ffcc99', '#ff9999'])
plt.title('Proportions of {x.title()} (Matplotlib)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()""",

        "line_chart": """import matplotlib.pyplot as plt

# Ordered comparison across parental education
edu_order = ["some high school", "high school", "some college", 
             "associate's degree", "bachelor's degree", "master's degree"]
grouped = df.groupby('parental level of education')[['math score', 'reading score']].mean().reindex(edu_order)

plt.figure(figsize=(8, 5))
plt.plot(grouped.index, grouped['math score'], marker='o', label='Math', color='blue')
plt.plot(grouped.index, grouped['reading score'], marker='s', label='Reading', color='green')
plt.title('Mean Scores across Education (Matplotlib)', fontsize=14, fontweight='bold')
plt.xlabel('Parental Education Level')
plt.ylabel('Average Score')
plt.xticks(rotation=20, ha='right')
plt.legend()
plt.tight_layout()
plt.show()""",

        "pairplot": """import matplotlib.pyplot as plt
import pandas as pd

# Matplotlib doesn't have an automatic pairplot function.
# You have to use pandas.plotting.scatter_matrix or create subplots manually:
from pandas.plotting import scatter_matrix

scatter_matrix(df[['math score', 'reading score', 'writing score']],
               figsize=(8, 8), diagonal='kde', color='royalblue')
plt.suptitle('Pairwise Scatter Matrix (Matplotlib / Pandas)', y=0.92, fontweight='bold')
plt.show()"""
    }
    
    return codes.get(chart_type, "# Matplotlib code not available for this chart type.")


# ==============================================================================
# 17. GENERATE SEABORN CODE
# ==============================================================================
def generate_seaborn_code(chart_type, x_col=None, y_col=None, hue_col=None):
    """
    Generates beginner-friendly Seaborn code for the specified chart.
    """
    x = x_col or "math score"
    y = y_col or "reading score"
    hue = hue_col or "gender"
    
    codes = {
        "histogram": f"""import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

# Seaborn handles bins and Kernel Density Estimate (KDE) curve in one line!
sns.histplot(data=df, x='{x}', kde=True, bins=20, color='teal')

plt.title('Distribution of {x.title()} with KDE (Seaborn)', fontsize=14, fontweight='bold')
plt.xlabel('{x.title()}', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.show()""",

        "bar_chart": f"""import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

# Seaborn automatically computes the mean and error bars!
sns.barplot(data=df, x='{x}', y='{y}', palette='Blues_d')

plt.title('Average {y.title()} by {x.title()} (Seaborn)', fontsize=14, fontweight='bold')
plt.xlabel('{x.title()}', fontsize=12)
plt.ylabel('Average {y.title()}', fontsize=12)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.show()""",

        "box_plot": f"""import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

# Pass DataFrame and column names directly; Seaborn handles grouping and styling
sns.boxplot(data=df, x='{y_col or "gender"}', y='{x}', palette='Set2')

plt.title('Box Plot of {x.title()} by {y_col or "Gender"} (Seaborn)', fontsize=14, fontweight='bold')
plt.xlabel('{(y_col or "gender").title()}', fontsize=12)
plt.ylabel('{x.title()}', fontsize=12)
plt.tight_layout()
plt.show()""",

        "scatter_plot": f"""import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

# Seaborn adds hue categorization and clean legends automatically
sns.scatterplot(data=df, x='{x}', y='{y}', hue='{hue}', palette='tab10', alpha=0.7)

# Add linear regression fit line
sns.regplot(data=df, x='{x}', y='{y}', scatter=False, color='red')

plt.title('{x.title()} vs {y.title()} by {hue.title()} (Seaborn)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()""",

        "heatmap": """import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))

# Seaborn computes annotations, colormap, and colorbar in a single call!
numeric_df = df[['math score', 'reading score', 'writing score']]
sns.heatmap(numeric_df.corr(), annot=True, cmap='YlGnBu', vmin=0.5, vmax=1.0, fmt='.2f')

plt.title('Correlation Heatmap (Seaborn)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()""",

        "count_plot": f"""import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

# Dedicated function for frequency counting categorical features
sns.countplot(data=df, x='{x}', palette='crest')

plt.title('Count of Students by {x.title()} (Seaborn)', fontsize=14, fontweight='bold')
plt.xlabel('{x.title()}', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.show()""",

        "pie_chart": f"""# Note: Seaborn specializes in statistical plots and does not have a dedicated pie chart function.
# In Python data science, we use Matplotlib directly for pie charts:
import matplotlib.pyplot as plt

counts = df['{x}'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of {x.title()}', fontsize=14, fontweight='bold')
plt.show()""",

        "line_chart": """import seaborn as sns
import matplotlib.pyplot as plt

# Seaborn lineplot handles grouping and aggregated mean calculation automatically
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='parental level of education', y='math score', marker='o', color='teal')

plt.title('Math Score Trend across Education Levels (Seaborn)', fontsize=14, fontweight='bold')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.show()""",

        "pairplot": f"""import seaborn as sns
import matplotlib.pyplot as plt

# A signature feature of Seaborn: one line generates complete pairwise grid
sns.pairplot(df[['math score', 'reading score', 'writing score', '{hue}']],
             hue='{hue}', palette='tab10', diag_kind='kde')

plt.show()"""
    }
    
    return codes.get(chart_type, "# Seaborn code not available for this chart type.")


# ==============================================================================
# 18. CREATE DASHBOARD HTML
# ==============================================================================
def create_dashboard_html():
    """
    Returns the complete HTML, CSS, and JavaScript single-page application.
    Clean, responsive classroom teacher aesthetic without external bloat.
    """
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EDA Classroom Dashboard | Students Performance</title>
  <style>
    :root {
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
      --primary-light: #dbeafe;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #1e293b;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --code-bg: #0f172a;
      --radius: 8px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    header {
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      color: white;
      padding: 1.25rem 2rem;
      border-bottom: 3px solid var(--primary);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }
    .header-title h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.5px; }
    .header-title p { font-size: 0.875rem; color: #94a3b8; }
    .badge {
      display: inline-block;
      padding: 0.25rem 0.6rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      background: rgba(37, 99, 235, 0.25);
      color: #93c5fd;
      border: 1px solid rgba(147, 197, 253, 0.3);
    }
    nav.tabs {
      background: white;
      border-bottom: 1px solid var(--border);
      padding: 0 2rem;
      display: flex;
      gap: 0.5rem;
      overflow-x: auto;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .tab-btn {
      background: none;
      border: none;
      padding: 0.9rem 1.25rem;
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      border-bottom: 3px solid transparent;
      transition: all 0.2s;
      white-space: nowrap;
    }
    .tab-btn:hover { color: var(--primary); }
    .tab-btn.active {
      color: var(--primary);
      border-bottom-color: var(--primary);
      background-color: var(--primary-light);
    }
    main { max-width: 1280px; margin: 1.5rem auto; padding: 0 1.5rem; }
    .tab-content { display: none; }
    .tab-content.active { display: block; animation: fadeIn 0.25s ease-in; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    /* Teacher Callout Card */
    .teacher-box {
      background: #eff6ff;
      border-left: 4px solid var(--primary);
      padding: 1rem 1.25rem;
      border-radius: 0 var(--radius) var(--radius) 0;
      margin-bottom: 1.5rem;
    }
    .teacher-box h3 { font-size: 1rem; color: #1e40af; margin-bottom: 0.25rem; display: flex; align-items: center; gap: 0.5rem; }
    .teacher-box p { font-size: 0.875rem; color: #1e3a8a; }

    /* Metric Cards Grid */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    .stat-card {
      background: var(--card-bg);
      padding: 1.25rem;
      border-radius: var(--radius);
      border: 1px solid var(--border);
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stat-card .label { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: var(--text-muted); }
    .stat-card .value { font-size: 1.75rem; font-weight: 700; color: var(--text); margin-top: 0.25rem; }
    .stat-card .subtext { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem; }

    /* Card Panels */
    .card {
      background: var(--card-bg);
      border-radius: var(--radius);
      border: 1px solid var(--border);
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.5rem;
    }

    /* Controls Bar */
    .controls-row {
      display: flex;
      gap: 1rem;
      align-items: flex-end;
      flex-wrap: wrap;
      background: #f1f5f9;
      padding: 1rem;
      border-radius: var(--radius);
      margin-bottom: 1.25rem;
    }
    .form-group { display: flex; flex-direction: column; gap: 0.35rem; }
    .form-group label { font-size: 0.8rem; font-weight: 600; color: var(--text); }
    select, button {
      padding: 0.5rem 0.85rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      font-size: 0.9rem;
      background: white;
      color: var(--text);
    }
    select:focus { outline: 2px solid var(--primary); }
    button.btn-primary {
      background: var(--primary);
      color: white;
      border: none;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }
    button.btn-primary:hover { background: var(--primary-dark); }
    button.btn-outline {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text);
      cursor: pointer;
    }
    button.btn-outline:hover { background: #e2e8f0; }

    /* Visualizer Container */
    .visualizer-grid {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 1.5rem;
      align-items: start;
    }
    @media (max-width: 960px) {
      .visualizer-grid { grid-template-columns: 1fr; }
    }
    .chart-display {
      background: white;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1rem;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 420px;
    }
    .chart-display img {
      max-width: 100%;
      height: auto;
      border-radius: 4px;
    }

    /* Explanation Cards */
    .explanation-panel {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .info-card {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1rem;
    }
    .info-card h4 {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--primary);
      margin-bottom: 0.4rem;
      font-weight: 700;
    }
    .info-card p { font-size: 0.875rem; color: #334155; }

    /* Code View Section */
    .code-accordion {
      margin-top: 1.25rem;
      border-top: 1px solid var(--border);
      padding-top: 1rem;
    }
    .code-tabs {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }
    .code-tab-btn {
      padding: 0.35rem 0.75rem;
      font-size: 0.8rem;
      font-weight: 600;
      border-radius: 4px;
      border: 1px solid var(--border);
      background: #f1f5f9;
      cursor: pointer;
    }
    .code-tab-btn.active {
      background: var(--code-bg);
      color: #38bdf8;
      border-color: var(--code-bg);
    }
    pre {
      background: var(--code-bg);
      color: #f8fafc;
      padding: 1rem;
      border-radius: 6px;
      overflow-x: auto;
      font-family: Consolas, Monaco, "Courier New", monospace;
      font-size: 0.85rem;
      line-height: 1.45;
    }

    /* Comparison Table */
    .table-container {
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: var(--radius);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
      text-align: left;
    }
    th {
      background: #f1f5f9;
      padding: 0.75rem 1rem;
      font-weight: 700;
      color: var(--text);
      border-bottom: 2px solid var(--border);
      white-space: nowrap;
    }
    td {
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }
    tr:nth-child(even) { background-color: #fafbfc; }
    tr:hover { background-color: #f1f5f9; }

    /* Side by side code viewer */
    .code-compare-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-top: 1rem;
    }
    @media (max-width: 800px) {
      .code-compare-grid { grid-template-columns: 1fr; }
    }
    .code-compare-box {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
    }
    .code-compare-header {
      background: #1e293b;
      color: white;
      padding: 0.5rem 1rem;
      font-weight: 600;
      font-size: 0.85rem;
      display: flex;
      justify-content: space-between;
    }

    /* Error Banner */
    .error-banner {
      background: #fee2e2;
      border-left: 4px solid var(--danger);
      padding: 1.25rem;
      color: #991b1b;
      border-radius: var(--radius);
      margin-bottom: 1.5rem;
    }
    .error-banner h3 { font-size: 1.1rem; margin-bottom: 0.5rem; }
    .error-banner ol { margin-left: 1.5rem; margin-top: 0.5rem; font-size: 0.9rem; }

    /* Loading Spinner */
    .spinner {
      border: 3px solid rgba(0, 0, 0, 0.1);
      width: 32px;
      height: 32px;
      border-radius: 50%;
      border-left-color: var(--primary);
      animation: spin 0.8s linear infinite;
      margin: 2rem auto;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <header>
    <div class="header-title">
      <h1>Single-File Interactive EDA Dashboard</h1>
      <p>Classroom Instructor Edition &bull; Python Data Visualization &amp; Analysis</p>
    </div>
    <div>
      <span class="badge">StudentsPerformance.csv</span>
      <span class="badge" style="background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border-color: rgba(110, 231, 183, 0.3);">
        Matplotlib &amp; Seaborn
      </span>
    </div>
  </header>

  <!-- NAVIGATION TABS -->
  <nav class="tabs">
    <button class="tab-btn active" onclick="switchTab('overview')">1. Overview &amp; EDA Intro</button>
    <button class="tab-btn" onclick="switchTab('univariate')">2. Univariate Analysis</button>
    <button class="tab-btn" onclick="switchTab('bivariate')">3. Bivariate Analysis</button>
    <button class="tab-btn" onclick="switchTab('multivariate')">4. Multivariate Analysis</button>
    <button class="tab-btn" onclick="switchTab('compare')">5. Matplotlib vs Seaborn</button>
    <button class="tab-btn" onclick="switchTab('cheatsheet')">6. Graph Comparison Table</button>
  </nav>

  <main>
    <!-- CSV Missing Error Container (Populated if file not found) -->
    <div id="error-container"></div>

    <!-- ===================================================================== -->
    <!-- TAB 1: OVERVIEW -->
    <!-- ===================================================================== -->
    <div id="tab-overview" class="tab-content active">
      <div class="teacher-box">
        <h3>Instructor Lecture: What is Exploratory Data Analysis (EDA)?</h3>
        <p>
          <strong>Exploratory Data Analysis (EDA)</strong> is the vital first step in any data science project.
          Coined by statistician John Tukey, EDA is the practice of investigating datasets to summarize their main characteristics,
          often using visual methods. Instead of jumping directly into statistical modeling or machine learning, EDA helps us:
          <strong>(1)</strong> Understand feature distributions and skewness,
          <strong>(2)</strong> Detect errors, missing values, and outliers,
          <strong>(3)</strong> Uncover relationships between variables, and
          <strong>(4)</strong> Formulate testable hypotheses.
        </p>
      </div>

      <!-- KEY METRIC CARDS -->
      <div class="stats-grid">
        <div class="stat-card">
          <div class="label">Total Students</div>
          <div class="value" id="card-total-students">--</div>
          <div class="subtext">1,000 Exam Candidates</div>
        </div>
        <div class="stat-card">
          <div class="label">Dataset Columns</div>
          <div class="value" id="card-total-cols">--</div>
          <div class="subtext">5 Categorical, 3 Numeric</div>
        </div>
        <div class="stat-card">
          <div class="label">Average Math Score</div>
          <div class="value" id="card-avg-math" style="color: #2563eb;">--</div>
          <div class="subtext">Scale: 0 - 100</div>
        </div>
        <div class="stat-card">
          <div class="label">Average Reading Score</div>
          <div class="value" id="card-avg-reading" style="color: #10b981;">--</div>
          <div class="subtext">Scale: 0 - 100</div>
        </div>
        <div class="stat-card">
          <div class="label">Average Writing Score</div>
          <div class="value" id="card-avg-writing" style="color: #f59e0b;">--</div>
          <div class="subtext">Scale: 0 - 100</div>
        </div>
        <div class="stat-card">
          <div class="label">Missing / Duplicates</div>
          <div class="value" id="card-clean-status">0 / 0</div>
          <div class="subtext">Clean Kaggle Dataset</div>
        </div>
      </div>

      <!-- DATASET PREVIEW TABLE -->
      <div class="card">
        <div class="card-title">
          <span>Dataset Preview (First 8 Rows)</span>
          <span style="font-size: 0.85rem; font-weight: normal; color: var(--text-muted);">
            Source: Kaggle Students' Performance in Exams
          </span>
        </div>
        <div class="table-container">
          <table id="preview-table">
            <thead>
              <tr>
                <th>gender</th>
                <th>race/ethnicity</th>
                <th>parental level of education</th>
                <th>lunch</th>
                <th>test preparation course</th>
                <th>math score</th>
                <th>reading score</th>
                <th>writing score</th>
              </tr>
            </thead>
            <tbody>
              <tr><td colspan="8" style="text-align:center;">Loading dataset preview...</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- STATISTICAL SUMMARY -->
      <div class="card">
        <div class="card-title">
          <span>Descriptive Statistics (Pandas .describe())</span>
        </div>
        <div class="table-container">
          <table id="stats-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Math Score</th>
                <th>Reading Score</th>
                <th>Writing Score</th>
              </tr>
            </thead>
            <tbody>
              <tr><td colspan="4" style="text-align:center;">Loading statistical summary...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 2: UNIVARIATE ANALYSIS -->
    <!-- ===================================================================== -->
    <div id="tab-univariate" class="tab-content">
      <div class="teacher-box">
        <h3>Classroom Guide: Univariate Analysis</h3>
        <p>
          <strong>Univariate analysis</strong> examines <em>one single variable at a time</em>. Its goal is to describe
          distribution, frequency, central tendency (mean, median, mode), and spread (standard deviation, IQR, outliers).
          For <strong>numeric columns</strong>, use <em>Histograms</em> and <em>Box Plots</em>.
          For <strong>categorical columns</strong>, use <em>Count Plots</em> and <em>Pie Charts</em>.
        </p>
      </div>

      <div class="card">
        <div class="controls-row">
          <div class="form-group">
            <label for="uni-col-select">Select Column:</label>
            <select id="uni-col-select" onchange="onUnivariateColChange()">
              <!-- Options populated dynamically -->
            </select>
          </div>
          <div class="form-group">
            <label for="uni-chart-select">Chart Type:</label>
            <select id="uni-chart-select">
              <!-- Dynamically updated based on numeric vs categorical -->
            </select>
          </div>
          <button class="btn-primary" onclick="loadUnivariateChart()">Render Chart</button>
        </div>

        <div class="visualizer-grid">
          <div class="chart-display" id="uni-chart-container">
            <div class="spinner"></div>
          </div>
          <div class="explanation-panel">
            <div class="info-card">
              <h4 id="uni-info-title">Chart Explanation</h4>
              <p id="uni-info-purpose">Select a column and chart type to view pedagogical notes.</p>
            </div>
            <div class="info-card">
              <h4>When to Use</h4>
              <p id="uni-info-when">--</p>
            </div>
            <div class="info-card">
              <h4>What Students Should Observe</h4>
              <p id="uni-info-observe">--</p>
            </div>
            <div class="info-card">
              <h4>Feature Data Type</h4>
              <p id="uni-info-datatype">--</p>
            </div>
          </div>
        </div>

        <!-- CODE GENERATION -->
        <div class="code-accordion">
          <div class="code-tabs">
            <button class="code-tab-btn active" id="uni-tab-seaborn" onclick="toggleCodeView('uni', 'seaborn')">Seaborn Code</button>
            <button class="code-tab-btn" id="uni-tab-matplotlib" onclick="toggleCodeView('uni', 'matplotlib')">Matplotlib Code</button>
            <button class="btn-outline" style="margin-left: auto; padding: 0.2rem 0.6rem; font-size: 0.75rem;" onclick="copyCode('uni')">Copy Code</button>
          </div>
          <pre><code id="uni-code-display"># Python code will appear here after rendering</code></pre>
        </div>
      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 3: BIVARIATE ANALYSIS -->
    <!-- ===================================================================== -->
    <div id="tab-bivariate" class="tab-content">
      <div class="teacher-box">
        <h3>Classroom Guide: Bivariate Analysis</h3>
        <p>
          <strong>Bivariate analysis</strong> analyzes the relationship between <em>two variables simultaneously</em>.
          It tests hypotheses about dependencies, correlations, and group discrepancies.
          Common patterns:
          <strong>(1) Numeric vs Numeric</strong> &rarr; <em>Scatter Plot</em> (correlation).
          <strong>(2) Categorical vs Numeric</strong> &rarr; <em>Bar Chart (means)</em> or <em>Box Plot by Category</em> (distribution comparison).
          <strong>(3) Ordered Progression</strong> &rarr; <em>Line Chart</em>.
        </p>
      </div>

      <div class="card">
        <div class="controls-row">
          <div class="form-group">
            <label for="bi-chart-type">Analysis Type:</label>
            <select id="bi-chart-type" onchange="onBivariateTypeChange()">
              <option value="scatter_plot">Scatter Plot (Numeric X vs Numeric Y)</option>
              <option value="bar_chart">Bar Chart (Category X vs Mean Score Y)</option>
              <option value="box_plot">Box Plot by Category (Category X vs Score Y)</option>
              <option value="line_chart">Line Chart (Ordered Education vs Subject Averages)</option>
            </select>
          </div>
          <div class="form-group" id="bi-x-group">
            <label for="bi-x-select">X Column:</label>
            <select id="bi-x-select"></select>
          </div>
          <div class="form-group" id="bi-y-group">
            <label for="bi-y-select">Y Column:</label>
            <select id="bi-y-select"></select>
          </div>
          <div class="form-group" id="bi-hue-group">
            <label for="bi-hue-select">Optional Grouping (Hue):</label>
            <select id="bi-hue-select">
              <option value="">(None)</option>
              <option value="gender">gender</option>
              <option value="lunch">lunch</option>
              <option value="test preparation course">test preparation course</option>
              <option value="race/ethnicity">race/ethnicity</option>
            </select>
          </div>
          <button class="btn-primary" onclick="loadBivariateChart()">Render Bivariate Plot</button>
        </div>

        <div class="visualizer-grid">
          <div class="chart-display" id="bi-chart-container">
            <div class="spinner"></div>
          </div>
          <div class="explanation-panel">
            <div class="info-card">
              <h4 id="bi-info-title">Chart Explanation</h4>
              <p id="bi-info-purpose">Select variables above to explore relationships.</p>
            </div>
            <div class="info-card">
              <h4>When to Use</h4>
              <p id="bi-info-when">--</p>
            </div>
            <div class="info-card">
              <h4>What Students Should Observe</h4>
              <p id="bi-info-observe">--</p>
            </div>
            <div class="info-card">
              <h4>Feature Data Types</h4>
              <p id="bi-info-datatype">--</p>
            </div>
          </div>
        </div>

        <!-- CODE GENERATION -->
        <div class="code-accordion">
          <div class="code-tabs">
            <button class="code-tab-btn active" id="bi-tab-seaborn" onclick="toggleCodeView('bi', 'seaborn')">Seaborn Code</button>
            <button class="code-tab-btn" id="bi-tab-matplotlib" onclick="toggleCodeView('bi', 'matplotlib')">Matplotlib Code</button>
            <button class="btn-outline" style="margin-left: auto; padding: 0.2rem 0.6rem; font-size: 0.75rem;" onclick="copyCode('bi')">Copy Code</button>
          </div>
          <pre><code id="bi-code-display"># Python code will appear here after rendering</code></pre>
        </div>
      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 4: MULTIVARIATE ANALYSIS -->
    <!-- ===================================================================== -->
    <div id="tab-multivariate" class="tab-content">
      <div class="teacher-box">
        <h3>Classroom Guide: Multivariate Analysis</h3>
        <p>
          <strong>Multivariate analysis</strong> studies interactions across <em>three or more variables at the same time</em>.
          It helps detect confounding variables, interaction effects, and high-dimensional clustering.
          Key tools include the <strong>Correlation Matrix Heatmap</strong> (matrix of Pearson <em>r</em> values) and
          the <strong>Pair Plot Matrix</strong> (pairwise scatter plots combined with diagonal density estimates).
        </p>
      </div>

      <div class="card">
        <div class="controls-row">
          <div class="form-group">
            <label for="multi-chart-type">Multivariate Method:</label>
            <select id="multi-chart-type" onchange="onMultivariateTypeChange()">
              <option value="heatmap">Correlation Matrix Heatmap (All Subjects)</option>
              <option value="pairplot">Pair Plot Grid (Math, Reading, Writing + Hue)</option>
            </select>
          </div>
          <div class="form-group" id="multi-hue-group">
            <label for="multi-hue-select">Color Grouping (Hue):</label>
            <select id="multi-hue-select">
              <option value="gender">gender</option>
              <option value="test preparation course">test preparation course</option>
              <option value="lunch">lunch</option>
              <option value="race/ethnicity">race/ethnicity</option>
            </select>
          </div>
          <button class="btn-primary" onclick="loadMultivariateChart()">Render Multivariate Plot</button>
        </div>

        <div class="visualizer-grid">
          <div class="chart-display" id="multi-chart-container">
            <div class="spinner"></div>
          </div>
          <div class="explanation-panel">
            <div class="info-card">
              <h4 id="multi-info-title">Multivariate Insights</h4>
              <p id="multi-info-purpose">--</p>
            </div>
            <div class="info-card">
              <h4>When to Use</h4>
              <p id="multi-info-when">--</p>
            </div>
            <div class="info-card">
              <h4>What Students Should Observe</h4>
              <p id="multi-info-observe">--</p>
            </div>
          </div>
        </div>

        <!-- CODE GENERATION -->
        <div class="code-accordion">
          <div class="code-tabs">
            <button class="code-tab-btn active" id="multi-tab-seaborn" onclick="toggleCodeView('multi', 'seaborn')">Seaborn Code</button>
            <button class="code-tab-btn" id="multi-tab-matplotlib" onclick="toggleCodeView('multi', 'matplotlib')">Matplotlib Code</button>
            <button class="btn-outline" style="margin-left: auto; padding: 0.2rem 0.6rem; font-size: 0.75rem;" onclick="copyCode('multi')">Copy Code</button>
          </div>
          <pre><code id="multi-code-display"># Python code will appear here after rendering</code></pre>
        </div>
      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 5: MATPLOTLIB VS SEABORN -->
    <!-- ===================================================================== -->
    <div id="tab-compare" class="tab-content">
      <div class="teacher-box">
        <h3>Instructor Lecture: Understanding Matplotlib vs Seaborn</h3>
        <p>
          Students often ask: <em>"Should I use Matplotlib or Seaborn?"</em> The answer is: <strong>both, because Seaborn is built directly on top of Matplotlib!</strong>
        </p>
        <ul style="margin-left: 1.5rem; margin-top: 0.5rem; font-size: 0.875rem;">
          <li><strong>Matplotlib (Low-level "Canvas &amp; Brushes"):</strong> Gives you total pixel-by-pixel control over figures, subplots, ticks, and legends. However, statistical operations (like calculating means, standard deviations, or box plot quartiles for groups) must often be done manually.</li>
          <li><strong>Seaborn (High-level "Statistical Grammar"):</strong> Designed specifically for Pandas DataFrames. It automatically computes means, standard deviations, confidence intervals, and color groupings (hue) in concise one-line functions.</li>
          <li><strong>Teacher's Rule of Thumb:</strong> Start exploratory data analysis with Seaborn. Use Matplotlib to customize the layout, fine-tune axes, or create bespoke visual annotations.</li>
        </ul>
      </div>

      <div class="card">
        <div class="controls-row">
          <div class="form-group">
            <label for="code-compare-select">Select Graph Type for Side-by-Side Code Comparison:</label>
            <select id="code-compare-select" onchange="renderCodeComparison()">
              <option value="histogram">1. Histogram</option>
              <option value="bar_chart">2. Bar Chart</option>
              <option value="box_plot">3. Box Plot</option>
              <option value="scatter_plot">4. Scatter Plot</option>
              <option value="heatmap">5. Correlation Heatmap</option>
            </select>
          </div>
        </div>

        <div id="code-compare-pedagogy" style="margin-bottom: 1rem; font-size: 0.9rem; color: #334155; background: #f8fafc; padding: 0.75rem 1rem; border-radius: 6px; border: 1px solid var(--border);">
          <!-- Dynamic pedagogical note -->
        </div>

        <div class="code-compare-grid">
          <div class="code-compare-box">
            <div class="code-compare-header">
              <span>Matplotlib Implementation</span>
              <button class="btn-outline" style="color: white; border-color: #475569; padding: 0.1rem 0.5rem; font-size: 0.75rem;" onclick="copyText('compare-mpl-code')">Copy</button>
            </div>
            <pre><code id="compare-mpl-code"># Matplotlib code</code></pre>
          </div>
          <div class="code-compare-box">
            <div class="code-compare-header">
              <span>Seaborn Implementation</span>
              <button class="btn-outline" style="color: white; border-color: #475569; padding: 0.1rem 0.5rem; font-size: 0.75rem;" onclick="copyText('compare-sns-code')">Copy</button>
            </div>
            <pre><code id="compare-sns-code"># Seaborn code</code></pre>
          </div>
        </div>
      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- TAB 6: GRAPH COMPARISON TABLE -->
    <!-- ===================================================================== -->
    <div id="tab-cheatsheet" class="tab-content">
      <div class="teacher-box">
        <h3>Classroom Reference: Visualization Decision Matrix</h3>
        <p>
          Knowing <em>which</em> chart to pick based on question type and data types is the hallmark of an effective data analyst.
          Use this reference table when designing data science dashboards or exploratory reports.
        </p>
      </div>

      <div class="card">
        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Graph Name</th>
                <th>Primary Purpose</th>
                <th>When to Use</th>
                <th>Example Question Answered</th>
                <th>Input Data Types</th>
                <th>Key Limitations</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Histogram</strong></td>
                <td>Display distribution shape, frequency spread, and central peak of a continuous variable.</td>
                <td>Univariate analysis of continuous metrics; checking for normal vs skewed distribution.</td>
                <td><em>What is the distribution of math scores among high school students?</em></td>
                <td>1 Continuous Numerical</td>
                <td>Heavily sensitive to bin size; hides exact individual values.</td>
              </tr>
              <tr>
                <td><strong>Box Plot</strong></td>
                <td>Display five-number summary (Min, Q1, Median, Q3, Max) and flag outliers.</td>
                <td>Comparing score dispersion and identifying severe underperformers or top outliers across groups.</td>
                <td><em>Do students with standard lunch have higher median writing scores and fewer low outliers?</em></td>
                <td>1 Numerical (or 1 Numeric + 1 Category)</td>
                <td>Cannot detect bimodal distributions (two peaks look like one); does not show sample size.</td>
              </tr>
              <tr>
                <td><strong>Bar Chart</strong></td>
                <td>Compare aggregated metrics (mean, sum) across distinct discrete categories.</td>
                <td>Bivariate comparison when one variable is categorical and the other is numerical.</td>
                <td><em>Which parental education level achieves the highest average reading score?</em></td>
                <td>1 Categorical + 1 Numerical (aggregated)</td>
                <td>Can mislead if Y-axis doesn't start at zero; does not display distribution variance.</td>
              </tr>
              <tr>
                <td><strong>Count Plot</strong></td>
                <td>Display exact frequency of occurrences in discrete categories.</td>
                <td>Univariate categorical inspection; assessing class imbalance in demographic data.</td>
                <td><em>How many students completed the test preparation course vs none?</em></td>
                <td>1 Categorical</td>
                <td>Only shows counts, not relationships between different exam subjects.</td>
              </tr>
              <tr>
                <td><strong>Pie Chart</strong></td>
                <td>Display relative percentage shares of parts to a whole (summing to 100%).</td>
                <td>High-level proportion comparison when categories are few (&le; 4 categories).</td>
                <td><em>What proportion of the exam cohort is female vs male?</em></td>
                <td>1 Categorical (few unique classes)</td>
                <td>Human visual perception is poor at judging angles; terrible for &gt; 5 categories.</td>
              </tr>
              <tr>
                <td><strong>Scatter Plot</strong></td>
                <td>Reveal correlation, direction, clusters, and bivariate relationships between two continuous scores.</td>
                <td>Testing if higher performance in one subject predicts performance in another.</td>
                <td><em>Does high reading proficiency strongly correlate with high writing proficiency?</em></td>
                <td>2 Continuous Numerical (+ optional Hue)</td>
                <td>Points overlap heavily in large datasets (overplotting); correlation does not prove causation.</td>
              </tr>
              <tr>
                <td><strong>Line Chart</strong></td>
                <td>Display continuous progression, trends, or ordered sequence relationships.</td>
                <td>Tracking score progression across ordered categories or sorted percentiles.</td>
                <td><em>How do mean scores scale as parental education level progresses from high school to master's?</em></td>
                <td>1 Ordered Sequence X + 1+ Numerical Y</td>
                <td>Misleading if X-axis lacks natural ordering (implies continuity between unrelated categories).</td>
              </tr>
              <tr>
                <td><strong>Heatmap</strong></td>
                <td>Visualize matrix magnitudes and correlation coefficients using color intensity.</td>
                <td>Multivariate correlation check; identifying multicollinearity across all exam subjects.</td>
                <td><em>Which pair of subjects has the strongest statistical correlation?</em></td>
                <td>Multiple Numerical columns (Correlation Matrix)</td>
                <td>Only reflects linear associations; sensitive to choice of color scale.</td>
              </tr>
              <tr>
                <td><strong>Pair Plot</strong></td>
                <td>Matrix of pairwise scatter plots and diagonal distributions across all numeric features.</td>
                <td>Comprehensive multivariate survey; exploring clustering by categorical demographic groups.</td>
                <td><em>How do all three exam scores interact simultaneously when colored by gender?</em></td>
                <td>All Numerical features + 1 Categorical Hue</td>
                <td>Computationally expensive on large datasets; visual clutter with many features.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </main>

  <!-- JAVASCRIPT LOGIC -->
  <script>
    // Global cache for summary and codes
    let datasetSummary = null;
    let currentUniCode = { seaborn: "", matplotlib: "" };
    let currentBiCode = { seaborn: "", matplotlib: "" };
    let currentMultiCode = { seaborn: "", matplotlib: "" };

    // Tab Navigation
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

      const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
      if (targetBtn) targetBtn.classList.add('active');

      const targetContent = document.getElementById('tab-' + tabId);
      if (targetContent) targetContent.classList.add('active');

      if (tabId === 'compare') {
        renderCodeComparison();
      }
    }

    // Initialize Application
    window.addEventListener('DOMContentLoaded', async () => {
      await loadDatasetSummary();
      populateDropdowns();
      // Load initial charts
      loadUnivariateChart();
      loadBivariateChart();
      loadMultivariateChart();
    });

    // 1. Fetch Dataset Summary
    async function loadDatasetSummary() {
      try {
        const res = await fetch('/api/summary');
        const data = await res.json();

        if (!data.success) {
          showErrorBanner(data.error);
          return;
        }

        datasetSummary = data.summary;

        // Populate Overview Cards
        document.getElementById('card-total-students').innerText = datasetSummary.total_students.toLocaleString();
        document.getElementById('card-total-cols').innerText = datasetSummary.total_columns;
        document.getElementById('card-avg-math').innerText = datasetSummary.avg_math.toFixed(1);
        document.getElementById('card-avg-reading').innerText = datasetSummary.avg_reading.toFixed(1);
        document.getElementById('card-avg-writing').innerText = datasetSummary.avg_writing.toFixed(1);
        document.getElementById('card-clean-status').innerText = `${datasetSummary.duplicate_count} Dup / 0 Null`;

        // Render Preview Table
        renderPreviewTable(datasetSummary.preview);
        // Render Statistics Table
        renderStatsTable(datasetSummary.stats);
      } catch (err) {
        showErrorBanner("Could not connect to Flask API: " + err.message);
      }
    }

    function showErrorBanner(msg) {
      const banner = document.getElementById('error-container');
      banner.innerHTML = `
        <div class="error-banner">
          <h3>Dataset Error: StudentsPerformance.csv not loaded</h3>
          <p>${msg}</p>
          <ol>
            <li>Download <strong>StudentsPerformance.csv</strong> from Kaggle.</li>
            <li>Place it in the exact same directory as <code>eda_dashboard.py</code>.</li>
            <li>Refresh this browser page.</li>
          </ol>
        </div>
      `;
    }

    function renderPreviewTable(rows) {
      if (!rows || rows.length === 0) return;
      const tbody = document.querySelector('#preview-table tbody');
      tbody.innerHTML = rows.map(r => `
        <tr>
          <td><span class="badge" style="background:#e0e7ff; color:#3730a3;">${r['gender'] || ''}</span></td>
          <td>${r['race/ethnicity'] || ''}</td>
          <td>${r['parental level of education'] || ''}</td>
          <td>${r['lunch'] || ''}</td>
          <td>${r['test preparation course'] || ''}</td>
          <td><strong>${r['math score']}</strong></td>
          <td><strong>${r['reading score']}</strong></td>
          <td><strong>${r['writing score']}</strong></td>
        </tr>
      `).join('');
    }

    function renderStatsTable(stats) {
      if (!stats) return;
      const tbody = document.querySelector('#stats-table tbody');
      const metrics = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
      tbody.innerHTML = metrics.map(m => `
        <tr>
          <td><strong>${m.toUpperCase()}</strong></td>
          <td>${stats['math score'] && stats['math score'][m] !== undefined ? stats['math score'][m] : '--'}</td>
          <td>${stats['reading score'] && stats['reading score'][m] !== undefined ? stats['reading score'][m] : '--'}</td>
          <td>${stats['writing score'] && stats['writing score'][m] !== undefined ? stats['writing score'][m] : '--'}</td>
        </tr>
      `).join('');
    }

    // Populate Dynamic Dropdowns
    function populateDropdowns() {
      if (!datasetSummary) return;

      const numCols = datasetSummary.numeric_columns;
      const catCols = datasetSummary.categorical_columns;
      const allCols = datasetSummary.columns;

      // Univariate Column Select
      const uniSelect = document.getElementById('uni-col-select');
      uniSelect.innerHTML = `
        <optgroup label="Numerical Features">
          ${numCols.map(c => `<option value="${c}">${c}</option>`).join('')}
        </optgroup>
        <optgroup label="Categorical Features">
          ${catCols.map(c => `<option value="${c}">${c}</option>`).join('')}
        </optgroup>
      `;

      onUnivariateColChange();
      onBivariateTypeChange();
    }

    // Handle Univariate Column Change (Adapts Chart Choices)
    function onUnivariateColChange() {
      if (!datasetSummary) return;
      const col = document.getElementById('uni-col-select').value;
      const isNum = datasetSummary.numeric_columns.includes(col);
      const chartSelect = document.getElementById('uni-chart-select');

      if (isNum) {
        chartSelect.innerHTML = `
          <option value="histogram">Histogram (Distribution & KDE)</option>
          <option value="box_plot">Box Plot (Quartiles & Outliers)</option>
        `;
      } else {
        chartSelect.innerHTML = `
          <option value="count_plot">Count Plot (Frequencies)</option>
          <option value="pie_chart">Pie Chart (Proportions)</option>
          <option value="bar_chart">Bar Chart (Counts)</option>
        `;
      }
    }

    // Load Univariate Chart
    async function loadUnivariateChart() {
      const col = document.getElementById('uni-col-select').value;
      const chartType = document.getElementById('uni-chart-select').value;
      const container = document.getElementById('uni-chart-container');
      container.innerHTML = '<div class="spinner"></div>';

      try {
        const url = `/api/chart?type=${encodeURIComponent(chartType)}&x=${encodeURIComponent(col)}`;
        const res = await fetch(url);
        const data = await res.json();

        if (data.success) {
          container.innerHTML = `<img src="${data.image}" alt="${chartType} chart">`;
          document.getElementById('uni-info-title').innerText = data.explanation.title;
          document.getElementById('uni-info-purpose').innerText = data.explanation.purpose;
          document.getElementById('uni-info-when').innerText = data.explanation.when_to_use;
          document.getElementById('uni-info-observe').innerText = data.explanation.what_to_observe;
          document.getElementById('uni-info-datatype').innerText = data.explanation.data_types;

          currentUniCode.seaborn = data.seaborn_code;
          currentUniCode.matplotlib = data.matplotlib_code;
          toggleCodeView('uni', 'seaborn');
        } else {
          container.innerHTML = `<div style="color:red; padding:1rem;">${data.error}</div>`;
        }
      } catch (err) {
        container.innerHTML = `<div style="color:red; padding:1rem;">Error: ${err.message}</div>`;
      }
    }

    // Handle Bivariate Type Change
    function onBivariateTypeChange() {
      if (!datasetSummary) return;
      const type = document.getElementById('bi-chart-type').value;
      const xSelect = document.getElementById('bi-x-select');
      const ySelect = document.getElementById('bi-y-select');
      const numCols = datasetSummary.numeric_columns;
      const catCols = datasetSummary.categorical_columns;

      if (type === 'scatter_plot') {
        xSelect.innerHTML = numCols.map(c => `<option value="${c}" ${c==='math score'?'selected':''}>${c}</option>`).join('');
        ySelect.innerHTML = numCols.map(c => `<option value="${c}" ${c==='reading score'?'selected':''}>${c}</option>`).join('');
        document.getElementById('bi-hue-group').style.display = 'flex';
      } else if (type === 'bar_chart' || type === 'box_plot') {
        xSelect.innerHTML = catCols.map(c => `<option value="${c}" ${c==='parental level of education'?'selected':''}>${c}</option>`).join('');
        ySelect.innerHTML = numCols.map(c => `<option value="${c}" ${c==='math score'?'selected':''}>${c}</option>`).join('');
        document.getElementById('bi-hue-group').style.display = (type==='box_plot') ? 'none' : 'none';
      } else if (type === 'line_chart') {
        xSelect.innerHTML = `<option value="parental level of education">parental level of education (Ordered)</option>`;
        ySelect.innerHTML = numCols.map(c => `<option value="${c}">${c}</option>`).join('');
        document.getElementById('bi-hue-group').style.display = 'none';
      }
    }

    // Load Bivariate Chart
    async function loadBivariateChart() {
      const type = document.getElementById('bi-chart-type').value;
      const x = document.getElementById('bi-x-select').value;
      const y = document.getElementById('bi-y-select').value;
      const hue = document.getElementById('bi-hue-select').value;

      const container = document.getElementById('bi-chart-container');
      container.innerHTML = '<div class="spinner"></div>';

      try {
        let url = `/api/chart?type=${encodeURIComponent(type)}&x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}`;
        if (hue && type === 'scatter_plot') url += `&hue=${encodeURIComponent(hue)}`;

        const res = await fetch(url);
        const data = await res.json();

        if (data.success) {
          container.innerHTML = `<img src="${data.image}" alt="${type} chart">`;
          document.getElementById('bi-info-title').innerText = data.explanation.title;
          document.getElementById('bi-info-purpose').innerText = data.explanation.purpose;
          document.getElementById('bi-info-when').innerText = data.explanation.when_to_use;
          document.getElementById('bi-info-observe').innerText = data.explanation.what_to_observe;
          document.getElementById('bi-info-datatype').innerText = data.explanation.data_types;

          currentBiCode.seaborn = data.seaborn_code;
          currentBiCode.matplotlib = data.matplotlib_code;
          toggleCodeView('bi', 'seaborn');
        } else {
          container.innerHTML = `<div style="color:red; padding:1rem;">${data.error}</div>`;
        }
      } catch (err) {
        container.innerHTML = `<div style="color:red; padding:1rem;">Error: ${err.message}</div>`;
      }
    }

    // Handle Multivariate Type Change
    function onMultivariateTypeChange() {
      const type = document.getElementById('multi-chart-type').value;
      const hueGroup = document.getElementById('multi-hue-group');
      hueGroup.style.display = (type === 'pairplot') ? 'flex' : 'none';
    }

    // Load Multivariate Chart
    async function loadMultivariateChart() {
      const type = document.getElementById('multi-chart-type').value;
      const hue = document.getElementById('multi-hue-select').value;
      const container = document.getElementById('multi-chart-container');
      container.innerHTML = '<div class="spinner"></div>';

      try {
        let url = `/api/chart?type=${encodeURIComponent(type)}`;
        if (type === 'pairplot' && hue) url += `&hue=${encodeURIComponent(hue)}`;

        const res = await fetch(url);
        const data = await res.json();

        if (data.success) {
          container.innerHTML = `<img src="${data.image}" alt="${type} chart">`;
          document.getElementById('multi-info-title').innerText = data.explanation.title;
          document.getElementById('multi-info-purpose').innerText = data.explanation.purpose;
          document.getElementById('multi-info-when').innerText = data.explanation.when_to_use;
          document.getElementById('multi-info-observe').innerText = data.explanation.what_to_observe;

          currentMultiCode.seaborn = data.seaborn_code;
          currentMultiCode.matplotlib = data.matplotlib_code;
          toggleCodeView('multi', 'seaborn');
        } else {
          container.innerHTML = `<div style="color:red; padding:1rem;">${data.error}</div>`;
        }
      } catch (err) {
        container.innerHTML = `<div style="color:red; padding:1rem;">Error: ${err.message}</div>`;
      }
    }

    // Toggle Seaborn / Matplotlib code in visualizer panels
    function toggleCodeView(section, lib) {
      const codeCache = section === 'uni' ? currentUniCode : (section === 'bi' ? currentBiCode : currentMultiCode);
      const codeElement = document.getElementById(`${section}-code-display`);
      codeElement.innerText = codeCache[lib] || '# Code unavailable';

      document.getElementById(`${section}-tab-seaborn`).classList.toggle('active', lib === 'seaborn');
      document.getElementById(`${section}-tab-matplotlib`).classList.toggle('active', lib === 'matplotlib');
    }

    // Copy Code helper
    function copyCode(section) {
      const code = document.getElementById(`${section}-code-display`).innerText;
      navigator.clipboard.writeText(code).then(() => {
        alert("Python code copied to clipboard!");
      });
    }

    function copyText(id) {
      const text = document.getElementById(id).innerText;
      navigator.clipboard.writeText(text).then(() => {
        alert("Code snippet copied to clipboard!");
      });
    }

    // Render Tab 5: Side-by-Side Code Comparison
    async function renderCodeComparison() {
      const chartType = document.getElementById('code-compare-select').value;
      try {
        const res = await fetch(`/api/compare_code?type=${encodeURIComponent(chartType)}`);
        const data = await res.json();

        if (data.success) {
          document.getElementById('compare-mpl-code').innerText = data.matplotlib_code;
          document.getElementById('compare-sns-code').innerText = data.seaborn_code;
          document.getElementById('code-compare-pedagogy').innerHTML = `
            <strong>Instructor Note for ${chartType.replace('_', ' ').toUpperCase()}:</strong> ${data.pedagogy}
          `;
        }
      } catch (err) {
        console.error("Code comparison fetch error:", err);
      }
    }
  </script>
</body>
</html>"""
    return html_template


# ==============================================================================
# 19. CREATE FLASK ROUTES
# ==============================================================================
def create_flask_routes(app, df, csv_path, csv_error):
    """
    Attaches all Flask web routes and API endpoints to the application.
    """
    @app.route("/")
    def index():
        """Serves the single-page HTML dashboard."""
        return render_template_string(create_dashboard_html())
    
    @app.route("/api/summary")
    def api_summary():
        """Returns the dataset statistical summary as JSON."""
        if df is None:
            return jsonify({
                "success": False,
                "error": csv_error or "Dataset not loaded."
            }), 404
        
        summary = get_dataset_summary(df)
        return jsonify({
            "success": True,
            "summary": summary
        })
    
    @app.route("/api/chart")
    def api_chart():
        """
        Dynamically generates and returns chart image (base64), explanation,
        and beginner-friendly Python code snippets.
        """
        if df is None:
            return jsonify({"success": False, "error": csv_error or "Dataset not loaded."}), 404
            
        chart_type = request.args.get("type", "histogram").lower()
        x_col = request.args.get("x", "math score")
        y_col = request.args.get("y", None)
        hue_col = request.args.get("hue", None)
        
        # Security / validation: ensure columns exist
        if x_col and x_col not in df.columns and chart_type != "heatmap":
            x_col = "math score"
        if y_col and y_col not in df.columns:
            y_col = None
            
        try:
            image_b64 = None
            if chart_type == "histogram":
                image_b64 = create_histogram(df, x_col)
            elif chart_type == "bar_chart":
                image_b64 = create_bar_chart(df, x_col, y_col)
            elif chart_type == "line_chart":
                image_b64 = create_line_chart(df, x_col, y_col)
            elif chart_type == "box_plot":
                # For bivariate, pass category as x_col and score as y_col, or univariate
                if y_col and pd.api.types.is_numeric_dtype(df[y_col]):
                    image_b64 = create_box_plot(df, y_col, category_column=x_col)
                else:
                    image_b64 = create_box_plot(df, x_col, category_column=None)
            elif chart_type == "scatter_plot":
                y_col = y_col or "reading score"
                image_b64 = create_scatter_plot(df, x_col, y_col, hue_column=hue_col)
            elif chart_type == "heatmap":
                image_b64 = create_heatmap(df)
            elif chart_type == "count_plot":
                image_b64 = create_count_plot(df, x_col)
            elif chart_type == "pie_chart":
                image_b64 = create_pie_chart(df, x_col)
            elif chart_type == "pairplot":
                hue = hue_col or "gender"
                image_b64 = create_pairplot(df, hue_column=hue)
            else:
                return jsonify({"success": False, "error": f"Unknown chart type '{chart_type}'."}), 400
                
            explanation = get_chart_explanation(chart_type, x_col, y_col, hue_col)
            mpl_code = generate_matplotlib_code(chart_type, x_col, y_col, hue_col)
            sns_code = generate_seaborn_code(chart_type, x_col, y_col, hue_col)
            
            return jsonify({
                "success": True,
                "image": image_b64,
                "explanation": explanation,
                "matplotlib_code": mpl_code,
                "seaborn_code": sns_code
            })
            
        except Exception as e:
            return jsonify({"success": False, "error": f"Chart generation error: {str(e)}"}), 500
            
    @app.route("/api/compare_code")
    def api_compare_code():
        """Returns side-by-side Matplotlib and Seaborn code with teacher explanations."""
        chart_type = request.args.get("type", "histogram").lower()
        
        pedagogy_notes = {
            "histogram": "In Matplotlib, you explicitly specify bins, colors, and manually add vertical lines for mean/median. In Seaborn, sns.histplot(kde=True) calculates the smooth density curve automatically in a single call.",
            "bar_chart": "Matplotlib requires you to aggregate data first using df.groupby().mean(). Seaborn automatically calculates the mean and draws confidence interval error bars directly from raw data.",
            "box_plot": "In Matplotlib, grouped box plots require packing values into a list of arrays manually. Seaborn simply takes x='category' and y='score' and styles quartiles and outliers automatically.",
            "scatter_plot": "Matplotlib requires manual loops or color mappings to color points by category. Seaborn provides the 'hue' parameter which automatically creates color palettes and formatted legends.",
            "heatmap": "In Matplotlib, creating a correlation heatmap requires matshow(), manual colorbars, and nested for-loops to annotate numbers inside cells. Seaborn's sns.heatmap() does all of this in one parameter (annot=True)."
        }
        
        mpl_code = generate_matplotlib_code(chart_type)
        sns_code = generate_seaborn_code(chart_type)
        pedagogy = pedagogy_notes.get(chart_type, "Compare the syntax between Matplotlib and Seaborn.")
        
        return jsonify({
            "success": True,
            "chart_type": chart_type,
            "matplotlib_code": mpl_code,
            "seaborn_code": sns_code,
            "pedagogy": pedagogy
        })


# ==============================================================================
# 20. MAIN APPLICATION ENTRYPOINT
# ==============================================================================
def main():
    """
    Initializes and launches the interactive EDA Dashboard Flask server.
    """
    print("=" * 70)
    print("  Single-File Interactive EDA Dashboard (Students Performance)")
    print("=" * 70)
    
    # 1. Load and clean dataset
    print(f"Loading dataset from: {CSV_PATH} ...")
    raw_df, csv_error = load_dataset(CSV_PATH)
    
    if raw_df is not None:
        df = clean_dataset(raw_df)
        print(f"Dataset successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    else:
        df = None
        print(f"Warning: {csv_error}")
        print("The dashboard will run in setup mode and guide the user on loading the CSV.")
        
    # 2. Initialize Flask application
    app = Flask(__name__)
    create_flask_routes(app, df, CSV_PATH, csv_error)
    
    port = int(os.environ.get("PORT", 5000))
    print(f"\nDashboard server running at: http://127.0.0.1:{port}")
    print("Open the link above in your web browser to explore the dashboard.")
    print("Press Ctrl+C to terminate the server.\n" + "=" * 70)
    
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
