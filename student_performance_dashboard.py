
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("student_performance_data.csv")

subjects = [
    "Python", "DBMS", "Mathematics",
    "Operating_Systems", "Statistics"
]

print("\nORIGINAL DATA")
print(df.head())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS:", df.duplicated().sum())

df = df.drop_duplicates()

df[subjects] = df[subjects].fillna(0)
df[subjects] = df[subjects].clip(0, 100)

df["Total_Marks"] = df[subjects].sum(axis=1)
df["Average_Marks"] = df[subjects].mean(axis=1)

def calculate_grade(avg):
    if avg >= 90:
        return "A+"
    elif avg >= 80:
        return "A"
    elif avg >= 70:
        return "B"
    elif avg >= 60:
        return "C"
    elif avg >= 50:
        return "D"
    elif avg >= 40:
        return "E"
    else:
        return "F"

df["Grade"] = df["Average_Marks"].apply(calculate_grade)

df["Attendance_Percentage"] = (
    df["Classes_Attended"] / df["Classes_Held"]
) * 100

df["Attendance_Status"] = np.where(
    df["Attendance_Percentage"] >= 75,
    "Eligible",
    "Low Attendance"
)

df["Result"] = np.where(
    (df[subjects] >= 40).all(axis=1),
    "Pass",
    "Fail"
)

df["Rank"] = (
    df["Average_Marks"]
    .rank(method="dense", ascending=False)
    .astype(int)
)

df = df.sort_values(
    ["Rank", "Student_ID"]
).reset_index(drop=True)

print("\nSTUDENT PERFORMANCE")
print(df[[
    "Student_ID", "Student_Name", "Total_Marks",
    "Average_Marks", "Grade", "Result", "Rank",
    "Attendance_Percentage", "Attendance_Status"
]].to_string(index=False))

print("\nTOPPER")
print(df[df["Rank"] == 1][
    ["Student_ID", "Student_Name", "Average_Marks", "Grade"]
])

low_attendance = df[df["Attendance_Percentage"] < 75]

print("\nLOW ATTENDANCE STUDENTS")
print(low_attendance[[
    "Student_ID", "Student_Name", "Attendance_Percentage"
]].to_string(index=False))

total_students = len(df)
passed = (df["Result"] == "Pass").sum()
failed = (df["Result"] == "Fail").sum()
pass_percentage = (passed / total_students) * 100
class_average = df["Average_Marks"].mean()

print("\nCLASS SUMMARY")
print("Total Students:", total_students)
print("Passed:", passed)
print("Failed:", failed)
print("Pass Percentage:", round(pass_percentage, 2), "%")
print("Class Average:", round(class_average, 2))
print("Low Attendance:", len(low_attendance))

grade_order = ["A+", "A", "B", "C", "D", "E", "F"]

df["Grade"].value_counts().reindex(
    grade_order, fill_value=0
).plot(kind="bar", figsize=(8, 5))

plt.title("Student Grade Distribution")
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df["Attendance_Percentage"], bins=8, edgecolor="black")
plt.title("Attendance Distribution")
plt.xlabel("Attendance Percentage")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

top10 = df.nsmallest(10, "Rank").sort_values("Average_Marks")

plt.figure(figsize=(9, 6))
plt.barh(top10["Student_Name"], top10["Average_Marks"])
plt.title("Top 10 Students")
plt.xlabel("Average Marks")
plt.tight_layout()
plt.show()

df["Result"].value_counts().reindex(
    ["Pass", "Fail"], fill_value=0
).plot(kind="bar", figsize=(6, 5))

plt.title("Pass and Fail Count")
plt.xlabel("Result")
plt.ylabel("Students")
plt.tight_layout()
plt.show()

df.to_csv("student_dashboard_data.csv", index=False)
low_attendance.to_csv("low_attendance_students.csv", index=False)

print("\nFILES SAVED SUCCESSFULLY")
print("student_dashboard_data.csv")
print("low_attendance_students.csv")
