import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# โหลดข้อมูลจากไฟล์ CSV
data = pd.read_csv('data.csv')

# แปลงข้อมูลเพศ (Gender) และสถานะการสมรส (Marital_Status) เป็นตัวเลข
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Marital_Status'] = data['Marital_Status'].map({'Single': 0, 'Married': 1})

# ตรวจสอบว่าคอลัมน์ 'Status' ต้องการแปลงเป็นตัวเลขหรือไม่
if data['Status'].dtype == 'object':
    data['Status'] = data['Status'].astype('category').cat.codes  # แปลงเป็นตัวเลขอัตโนมัติ

# กำหนดฟีเจอร์ที่ใช้ทำนาย
X = data[['Age', 'Length_of_Service', 'Salary', 'Gender', 'Marital_Status']]

# เป้าหมาย (Target)
y = data['Status']

# แบ่งข้อมูลเป็นชุด train และ test (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# สร้างโมเดล Decision Tree
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=2)

# ฝึกโมเดล
dt_model.fit(X_train, y_train)

# ทำนายผล
y_pred = dt_model.predict(X_test)

# คำนวณความแม่นยำ
accuracy = (y_pred == y_test).mean()*100
print(f"Accuracy: {accuracy:.4f}")

# บันทึกโมเดลเป็นไฟล์ pkl
joblib.dump(dt_model, 'decision_tree_model.pkl')
print("โมเดล Decision Tree ถูกบันทึกลงไฟล์ 'decision_tree_model.pkl'")
