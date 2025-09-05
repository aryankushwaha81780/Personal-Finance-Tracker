
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def loadCsv(filepath : str) -> Tuple[List[str], List[str], np.ndarray]:
    df = pd.read_csv(filepath)
    months = df.iloc[:, 0].astype(str).tolist()
    categories = df.columns[1 : ].tolist()
    data = df.iloc[:, 1:].to_numpy(dtype = float)
    
    return months, categories, data

def generateSampleData(months: int = 12, seed: int = 42) -> Tuple[List[str], List[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    categories = ["Food", "Rent", "Transport", "Entertainment"]
    base = np.array([8000, 15000, 3000, 4000])

    monthIndex = np.arange(months).reshape(months, 1)
    trend = (monthIndex * np.array([50,10,5,20]))
    noise = rng.integers(-800, 1200, size=(months, len(categories)))
    data = base + trend + noise
    monthsNames = [f"2024-{i + 1:02d}" for i in range(months)]

    
    return monthsNames, categories, data.astype(float)


def categoryTotals(data: np.ndarray, categories: List[str]) -> Dict[str, float]:
    totals = data.sum(axis = 0)
    
    return dict(zip(categories, totals))

def monthlyTotals(data: np.ndarray, months: List[str]) -> Dict[str, float]:
    totals = data.sum(axis = 1)
    
    return dict(zip(months, totals))

def topCategory(totalsPerCategories: Dict[str, float]) -> Tuple[str,float]:
    cats = list(totalsPerCategories.keys())
    vals = np.array(list(totalsPerCategories.values()))
    idx = int(np.argmax(vals))
    
    return cats[idx], float(vals[idx])


def movingAverageForecast(series: np.ndarray, window: int = 3, nForecast: int = 3) -> np.ndarray:
    s = series.astype(float).copy()
    forecasts = []
    for _ in range(nForecast):
        if len(s) < window:
            nextVal = s.mean()
        else:
            nextVal = s[-window:].mean()
        
        forecasts.append(nextVal)
        s = np.append(s, nextVal)
    
    return np.array(forecasts)

def linearTrendForecast(series: np.ndarray, nForecast: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    x = np.arange(len(series))
    coeffs = np.polyfit(x, series, 1)   #  [slope, intercept]
    futureX = np.arange(len(series), len(series) + nForecast)
    forecast = np.polyval(coeffs, futureX)

    return forecast, coeffs


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0

    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def plotCategoryTrends(months: List[str], data: np.ndarray, categories: list[str]):
    totals = data.sum(axis = 1)
    plt.figure()
    plt.plot(months, totals, marker = 'o')
    plt.title("Monthly Total Expenses")
    plt.xlabel("Months")
    plt.ylabel("Total (INR)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    months, categories, data = generateSampleData(12)


    catTotal = categoryTotals(data, categories)
    monTotal = monthlyTotals(data, months)

    print("Category totals:")
    for c, v in catTotal.items():
        print(f" - {c}: ${int(v):,}")
    print("\nTop category:", topCategory(catTotal))


    maeForecasts = []
    for i, cat in enumerate(categories):
        historic = data[:, i]
        f = movingAverageForecast(historic, window = 3, nForecast = 3)
        maeForecasts.append(f)
    maeForecasts = np.vstack(maeForecasts).T

    print("\nMoving Average Forecasts (next 3 Months) per Category: ")
    for i, row in enumerate(maeForecasts, 1):
        print(f" Month+{i}: " + ", ".join(f"{c}: ${int(v):,}" for c, v in zip(categories, row)))
    

    totals = data.sum(axis = 1)
    linForecast, coeffs= linearTrendForecast(totals, nForecast = 3)
    print("\nLinear Trend Forecast (next 3 months) - totals:", [int(x) for x in linForecast])
    print("Trend slope:", coeffs[0])

    plotCategoryTrends(months, data, categories)