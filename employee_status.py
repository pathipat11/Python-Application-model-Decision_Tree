import pandas as pd
import joblib
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1300, 800)
        # MainWindow.showMaximized()
        # MainWindow.setStyleSheet("background-color: lightblue;")
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        MainWindow.setFont(font)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(700, 100, 450, 400))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setStyleSheet(
            "QTableWidget {"
            "border: 2px solid black;"
            "gridline-color: gray;"
            "selection-background-color: lightblue;"
            "}"
            "QHeaderView::section {"
            "background-color: lightgray;"
            "border: 1px solid black;"
            "padding: 4px;"
            "}"
        )

        
        # Labels and Input Fields
        self.lbTitle = QtWidgets.QLabel(self.centralwidget)
        self.lbTitle.setGeometry(QtCore.QRect(250, 20, 861, 61))
        font.setFamily("Kanit")
        font.setPointSize(22)
        font.setWeight(QtGui.QFont.Bold)
        self.lbTitle.setFont(font)
        self.lbTitle.setText("Predict Employee Status")
        self.lbTitle.setStyleSheet("color: navy; background-color: lightyellow; border-radius: 20px; padding: 10px;")
        self.lbTitle.setAlignment(QtCore.Qt.AlignCenter)

        self.lbAge = QtWidgets.QLabel(self.centralwidget)
        self.lbAge.setGeometry(QtCore.QRect(240, 100, 91, 61))
        font.setPointSize(18)
        font.setWeight(QtGui.QFont.Bold)
        self.lbAge.setFont(font)
        self.lbAge.setText("Age")

        self.textAge = QtWidgets.QTextEdit(self.centralwidget)
        self.textAge.setGeometry(QtCore.QRect(400, 100, 221, 51))
        font.setPointSize(14)
        self.textAge.setFont(font)
        self.textAge.setAlignment(QtCore.Qt.AlignCenter)

        self.lbLength = QtWidgets.QLabel(self.centralwidget)
        self.lbLength.setGeometry(QtCore.QRect(80, 170, 411, 61))
        self.lbLength.setFont(font)
        self.lbLength.setText("Length of Service (Year)")

        self.textLength = QtWidgets.QTextEdit(self.centralwidget)
        self.textLength.setGeometry(QtCore.QRect(400, 170, 221, 51))
        self.textLength.setFont(font)
        self.textLength.setAlignment(QtCore.Qt.AlignCenter)

        self.lbSalary = QtWidgets.QLabel(self.centralwidget)
        self.lbSalary.setGeometry(QtCore.QRect(220, 240, 121, 61))
        self.lbSalary.setFont(font)
        self.lbSalary.setText("Salary")

        self.textSalary = QtWidgets.QTextEdit(self.centralwidget)
        self.textSalary.setGeometry(QtCore.QRect(400, 240, 221, 51))
        self.textSalary.setFont(font)
        self.textSalary.setAlignment(QtCore.Qt.AlignCenter)

        self.lbGender = QtWidgets.QLabel(self.centralwidget)
        self.lbGender.setGeometry(QtCore.QRect(220, 320, 141, 61))
        self.lbGender.setFont(font)
        self.lbGender.setText("Gender")

        self.gbGender = QtWidgets.QGroupBox(self.centralwidget)
        self.gbGender.setGeometry(QtCore.QRect(400, 300, 221, 71))
        self.rdbtMale = QtWidgets.QRadioButton("Male", self.gbGender)
        self.rdbtMale.setGeometry(QtCore.QRect(10, 30, 81, 20))
        self.rdbtFemale = QtWidgets.QRadioButton("Female", self.gbGender)
        self.rdbtFemale.setGeometry(QtCore.QRect(110, 30, 91, 20))

        self.lbStatus = QtWidgets.QLabel(self.centralwidget)
        self.lbStatus.setGeometry(QtCore.QRect(160, 390, 251, 61))
        self.lbStatus.setFont(font)
        self.lbStatus.setText("Marital Status")

        self.gbStatus = QtWidgets.QGroupBox(self.centralwidget)
        self.gbStatus.setGeometry(QtCore.QRect(400, 380, 221, 71))
        self.rdbtSingle = QtWidgets.QRadioButton("Single", self.gbStatus)
        self.rdbtSingle.setGeometry(QtCore.QRect(10, 30, 101, 21))
        self.rdbtMarried = QtWidgets.QRadioButton("Married", self.gbStatus)
        self.rdbtMarried.setGeometry(QtCore.QRect(110, 30, 95, 20))

        self.btPredict = QtWidgets.QPushButton("Predict", self.centralwidget)
        self.btPredict.setGeometry(QtCore.QRect(400, 480, 221, 51))
        self.btPredict.setStyleSheet("background-color: darkgreen; color: white; font-size: 18px; border-radius: 10px;")
        self.btPredict.clicked.connect(self.predict)

        self.btReset = QtWidgets.QPushButton("Reset", self.centralwidget)
        self.btReset.setGeometry(QtCore.QRect(400, 540, 221, 51))
        self.btReset.setStyleSheet("background-color: darkred; color: white; font-size: 18px; border-radius: 10px;")
        self.btReset.clicked.connect(self.reset)

        self.lbResult = QtWidgets.QLabel(self.centralwidget)
        self.lbResult.setGeometry(QtCore.QRect(200, 610, 121, 61))
        self.lbResult.setFont(font)
        self.lbResult.setText("Result")

        self.tbResult = QtWidgets.QTextBrowser(self.centralwidget)
        self.tbResult.setGeometry(QtCore.QRect(400, 610, 221, 51))

        MainWindow.setCentralWidget(self.centralwidget)

        # Load Data
        self.load_model()
        self.load_csv_data()

    def load_model(self):
        try:
            # ใช้ joblib.load แทน pickle.load
            self.model = joblib.load('model/decision_tree_model.pkl')
        except Exception as e:
            self.tbResult.setText(f"Error loading model: {str(e)}")

    def predict(self):
        try:
            # ดึงข้อมูลจาก Text Edit
            age = float(self.textAge.toPlainText().strip().replace("\n", ""))
            length = float(self.textLength.toPlainText().strip().replace("\n", ""))
            salary = float(self.textSalary.toPlainText().strip().replace("\n", ""))
            gender = 1 if self.rdbtMale.isChecked() else 0
            status = 1 if self.rdbtSingle.isChecked() else 0

            # สร้าง DataFrame สำหรับการทำนาย
            input_data = pd.DataFrame([[age, length, salary, gender, status]], columns=['Age', 'Length_of_Service', 'Salary', 'Gender', 'Marital_Status'])
            
            # ทำนายผลจากโมเดล
            result = self.model.predict(input_data)[0]
            
            # แสดงผลลัพธ์
            if result == 1:
                self.tbResult.setText("Employed")
                self.tbResult.setStyleSheet("color: green;")
                self.tbResult.setAlignment(QtCore.Qt.AlignCenter)
            else:
                self.tbResult.setText("Resigned")
                self.tbResult.setStyleSheet("color: red;")
                self.tbResult.setAlignment(QtCore.Qt.AlignCenter)
        
        except Exception as e:
            self.tbResult.setText(f"Error in prediction: {str(e)}")

    def reset(self):
        self.textAge.clear()
        self.textLength.clear()
        self.textSalary.clear()
        self.rdbtMale.setChecked(False)
        self.rdbtFemale.setChecked(False)
        self.rdbtSingle.setChecked(False)
        self.rdbtMarried.setChecked(False)
        self.tbResult.clear()
        
        self.textAge.setAlignment(QtCore.Qt.AlignCenter)
        self.textLength.setAlignment(QtCore.Qt.AlignCenter)
        self.textSalary.setAlignment(QtCore.Qt.AlignCenter)
    def load_csv_data(self):
        try:
            # โหลดข้อมูลจากไฟล์ data.csv
            df = pd.read_csv('data/data.csv')
            self.show_csv_data(df)
        except Exception as e:
            self.tbResult.setText(f"Error loading CSV: {str(e)}")

    def show_csv_data(self, df):
        self.tableWidget.setRowCount(df.shape[0])
        self.tableWidget.setColumnCount(df.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(df.columns)

        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                self.tableWidget.setItem(row, col, QtWidgets.QTableWidgetItem(str(df.iat[row, col])))
        
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())