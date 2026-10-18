# 📊 Cyclistic Bike-Share Analysis  
### Google Data Analytics Capstone Project  
**Author:** Daryl Steen  
**Tools Used:** R, tidyverse, ggplot2, lubridate, Kaggle Notebooks

---

## 💼 Business Task

Cyclistic, a bike-share company based in Chicago, wants to increase its annual memberships. As a junior data analyst, my role was to analyze 12 months of historical ride data to uncover differences in behavior between **casual riders** and **annual members**. The goal is to inform marketing strategies that convert casual riders into long-term members.

---

## 🔍 Guiding Questions
- How do casual riders and members use the Cyclistic service differently?
- What patterns exist in trip duration, time of day, and weekday usage?
- Can we identify actionable opportunities to increase membership conversion?

---

## 🧰 Tools & Techniques
- **R** for data cleaning, processing, analysis, and visualization  
- **Tidyverse** for efficient data wrangling  
- **ggplot2** for insightful visualizations  
- **Kaggle Notebook** for reproducible, shareable analysis  
- **T-test** for statistical comparison between rider types  

---

## 📌 Key Findings

### 1. 🗓️ Ride Frequency by Weekday
- **Casual riders** peak on **Saturdays (409,685)** and **Sundays (332,636)**, reinforcing their use of the service for **weekend leisure**.
- **Members** are most active during the **workweek**, especially **Wednesdays (557,010)** and **Tuesdays (539,334)**, suggesting **weekday commuting**.

### 2. ⏱️ Ride Duration Differences
| User Type | Average Ride Length |
|-----------|---------------------|
| Casual    | 20.9 mins           |
| Member    | 12.1 mins           |

- Casual riders take longer trips and show higher variability (confirmed via boxplot with mean markers).
- A two-sample **t-test** confirmed this difference is statistically significant (p < 0.001).

### 3. 📅 Seasonal Ride Trends
- Both rider types peak in warmer months (June–September).
- The trend is **seasonally aligned**, but **members** consistently ride more than casuals throughout the year.

---

## 📈 Visualizations
- Figure 1: Number of Rides by Weekday and User Type  
- Figure 2: Ride Length Distribution with Mean Indicator  
- Figure 3: Avg Ride Duration by Weekday  
- Figure 4: Monthly Ride Volume Trends

> 🔗 Full notebook with visuals on [Kaggle](https://www.kaggle.com/darylhs)

---

## 🎯 Recommendations

1. **Weekend Membership Campaigns**  
   Target casual weekend riders with Friday–Sunday promotional offers.

2. **Ride-Time Rewards Program**  
   Offer discounts or perks after 100+ minutes of use in a month to encourage repeat casual use.

3. **Highlight Commuter Benefits**  
   Promote convenience, savings, and speed of access for workday travel to drive member sign-ups.

---

## 🧭 Project Phases Completed
- **Ask:** Identified business problem and key questions  
- **Prepare:** Aggregated 12 months of public ride data  
- **Process:** Cleaned, structured, and validated the dataset  
- **Analyze:** Used statistical and visual methods to uncover insights  
- **Share:** Presented clear, visual findings in R and Markdown  
- **Act:** Delivered data-driven recommendations to support marketing goals

---

## 📘 Reflection
This capstone project solidified my skills in:
- Real-world data exploration and storytelling
- Statistical validation using t-tests
- Visualization for stakeholder clarity
- Applying business thinking to data analytics

I’m excited to continue refining my skills in R, Python, and advanced analytics!

---

## 📫 Connect with Me

- **LinkedIn:** [Daryl Steen](https://www.linkedin.com/in/daryl-steen-752a9283/)  
- **Kaggle:** [@darylhs](https://www.kaggle.com/darylhs)  
- **Portfolio:** Coming soon!
