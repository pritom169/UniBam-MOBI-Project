# Visitor Traffic and Weather Correlation Analysis

## 📌 Overview
This project analyzes the correlation between **visitor traffic** and **weather conditions** in Bamberg, Germany. It aggregates visitor data across multiple zones and merges it with weather data (temperature, precipitation, and wind speed) to explore potential relationships.

## 🚀 How to Run the Analysis
### 1️⃣ Install Required Libraries
Make sure you have Python installed. Then, install the necessary dependencies:
```bash
pip install pandas seaborn matplotlib
```

### 2️⃣ Run the Python Script
Navigate to the script directory and execute the following command:
```bash
python correlation_visitor_temperature.py
```

### 3️⃣ Check the Output
- The **correlation heatmap** and **scatter plots** will be saved in:
  ```
  D:\Bamberg University\winter24_25\project\UniBam-MOBI-Project\figures\
  ```
- The **correlation matrix** will be printed in the console.

## 📊 Understanding the Heatmap

The **heatmap** visualizes the correlation between **visitor traffic** and **weather conditions** using **Pearson correlation coefficients**.

### 🔍 What Does the Heatmap Represent?
- Each **cell** shows the **correlation coefficient** between two variables.
- The **color intensity** represents the strength of the relationship:
  - 🔴 **Red (Positive Correlation)** → When one variable increases, the other also increases.
  - 🔵 **Blue (Negative Correlation)** → When one variable increases, the other decreases.
  - ⚪ **White / Neutral (0 Correlation)** → No strong relationship.

### 🔢 How to Interpret Correlation Values?
| **Correlation Value** | **Interpretation** |
|-----------------|----------------------|
| **+1.00** | Perfect Positive Correlation (Both variables increase together) |
| **0.50 to 0.99** | Strong Positive Correlation |
| **0.10 to 0.49** | Weak Positive Correlation |
| **0.00** | No Correlation |
| **-0.10 to -0.49** | Weak Negative Correlation |
| **-0.50 to -0.99** | Strong Negative Correlation |
| **-1.00** | Perfect Negative Correlation (One increases, the other decreases) |

### 📌 Example Interpretation from the Heatmap
| **Variable 1** | **Variable 2** | **Correlation** | **Meaning** |
|--------------|--------------|--------------|----------------|
| **Total Visitors** | **Max Temperature (°C)** | **0.70** | Strong positive correlation: More visitors on warmer days. |
| **Total Visitors** | **Precipitation (mm)** | **-0.50** | Moderate negative correlation: Rainy days lead to fewer visitors. |
| **Total Visitors** | **Max Wind Speed (m/s)** | **-0.30** | Weak negative correlation: High winds slightly reduce visitors. |

### 📊 Key Takeaways
- **Higher temperatures** may **increase** visitor traffic.
- **Rainy days** tend to **reduce** visitors.
- **Wind speed** has a **minor effect** on visitor numbers.

## 📂 Output Files
After running the script, the following plots will be saved in `figures/`:
- `correlation_heatmap.png` → Correlation heatmap
- `visitor_vs_temperature.png` → Visitors vs. Temperature
- `visitor_vs_wind_speed.png` → Visitors vs. Wind Speed


