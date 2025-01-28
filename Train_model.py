import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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

# รายชื่อโมเดลที่ใช้
models = {
    "K-Nearest Neighbors (kNN)": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=2),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naïve Bayes": GaussianNB(var_smoothing=2e-1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Artificial Neural Network (ANN)": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',max_iter=1000, random_state=42)
}

# เทรนและทดสอบแต่ละโมเดล
for name, model in models.items():
    model.fit(X_train, y_train)  # ฝึกโมเดล
    y_pred = model.predict(X_test)  # ทำนายผล
    accuracy = accuracy_score(y_test, y_pred) * 100  # แปลงเป็นเปอร์เซ็นต์
    print(f"{name} Accuracy: {accuracy:.2f}%")
