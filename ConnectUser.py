import sys
import socket
import pickle

from Crypto.Random import get_random_bytes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


from Client import end_login
from Communication import TcpBySize
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox, QRadioButton
)
from PyQt6.QtGui import QFont

comms = None
decrypt_key = None

# ---------------- Entry Window ----------------
class EntryWindow(QWidget):
    def __init__(self, sock):
        super().__init__()
        self.setWindowTitle("Welcome")
        self.resize(300, 150)
        self.sock = sock

        swap_key(self.sock)

        label = QLabel("Choose an option:")
        label.setFont(QFont("Arial", 12))

        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.open_login)

        self.signup_button = QPushButton("Sign Up")
        self.signup_button.clicked.connect(self.open_signup)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.login_button)
        layout.addWidget(self.signup_button)
        self.setLayout(layout)

    def open_login(self):
        self.login_window = LoginWindow(self.sock)
        self.login_window.show()
        self.close()

    def open_signup(self):
        self.signup_window = SignupWindow(self.sock)
        self.signup_window.show()
        self.close()

# ---------------- Login Window ----------------
class LoginWindow(QWidget):
    def __init__(self, sock):
        super().__init__()
        self.setWindowTitle("Login")
        self.resize(350, 250)
        self.sock = sock

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Email")


        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color:red; font-weight:bold;")

        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.login)

        self.forgot_button = QPushButton("Forgot Password?")
        self.forgot_button.clicked.connect(self.forgot_password)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Login"))
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_input)
        layout.addWidget(self.email_input)
        layout.addWidget(self.error_label)
        layout.addWidget(self.login_button)
        layout.addWidget(self.forgot_button)

        self.setLayout(layout)

    def login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()
        email = self.email_input.text().strip()

        if not username or not password or not email:
            self.error_label.setText("Fill all fields!")
            return
        elif len(username) > 8:
            self.error_label.setText("Username is too long!")
            return

        user_data = {"code": "LOGIN", "username": username, "password": password, "email": email}
        bytes_user_data = pickle.dumps(user_data)
        comms.send_with_size(bytes_user_data)

        response: dict = pickle.loads(comms.recv_by_size())
        if response and response["code"] == "LOGINF":
            self.error_label.setText("Error with login!")
            return

        self.close()
        end_login(comms, username)

    def forgot_password(self):
        self.forgot_window = ForgotPasswordWindow(self.sock)
        self.forgot_window.show()
        self.close()

# ---------------- Signup Window ----------------
class SignupWindow(QWidget):
    def __init__(self, sock):
        super().__init__()
        self.setWindowTitle("Sign Up")
        self.resize(350, 300)
        self.sock = sock

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Email")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm_input = QLineEdit()
        self.confirm_input.setPlaceholderText("Re-enter Password")
        self.confirm_input.setEchoMode(QLineEdit.EchoMode.Password)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color:red; font-weight:bold;")

        self.submit_button = QPushButton("Create Account")
        self.submit_button.clicked.connect(self.signup)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Sign Up"))
        layout.addWidget(self.username_input)
        layout.addWidget(self.email_input)
        layout.addWidget(self.password_input)
        layout.addWidget(self.confirm_input)
        layout.addWidget(self.error_label)
        layout.addWidget(self.submit_button)
        self.setLayout(layout)

    def signup(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        email = self.email_input.text()
        confirm = self.confirm_input.text()

        if not username or not password or not confirm or not email:
            self.error_label.setText("All fields are required!")
            return

        if password != confirm:
            self.error_label.setText("Passwords do not match!")
            return


        user_data = {"code": "SIGNUP", "username": username, "password": password, "email": email}
        bytes_user_data = pickle.dumps(user_data)
        print("died")
        comms.send_with_size(bytes_user_data)

        print("died")

        response: dict = pickle.loads(comms.recv_by_size())

        if response and response["code"] == "SIGNUPF":
            self.error_label.setText("Error with sign-up!")
            return

        QMessageBox.information(self, "Success", "Signup info sent to server. Please login.")
        self.login_window = LoginWindow(self.sock)
        self.login_window.show()
        self.close()

# ---------------- Forgot Password Window ----------------
class ForgotPasswordWindow(QWidget):
    def __init__(self, sock):
        super().__init__()
        self.sock = sock

        self.setWindowTitle("Forgot Password")
        self.resize(350, 200)

        # ---- Widgets ----
        self.label = QLabel("Enter your email:")
        self.label.setFont(QFont("Arial", 12))

        self.email_input = QLineEdit()
        self.email_input.setFont(QFont("Arial", 12))
        self.email_input.setPlaceholderText("example@email.com")

        self.send_button = QPushButton("Send Code")
        self.send_button.setFont(QFont("Arial", 12))
        self.send_button.clicked.connect(self.send_code)

        self.back_button = QPushButton("Back to Login")
        self.back_button.setFont(QFont("Arial", 12))
        self.back_button.clicked.connect(self.back_to_login)

        # ---- Layout ----
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.email_input)
        layout.addWidget(self.send_button)
        layout.addWidget(self.back_button)

        self.setLayout(layout)

    def send_code(self):
        email = self.email_input.text().strip()

        if not email:
            QMessageBox.critical(self, "Error", "Email cannot be empty!")
            return

        self.enter_code_window = EnterCodeWindow(sock=self.sock, email=email)
        self.enter_code_window.show()
        self.close()

        comms.send_with_size({"code": "SEND_CODE", "email": email})

        print("Sending reset code to:", email)

        QMessageBox.information(self, "Success", "Reset code sent!")

    def back_to_login(self):
        self.close()
        self.login_window = LoginWindow(self.sock)
        self.login_window.show()

class ResetPasswordWindow(QWidget):
    def __init__(self, sock, email):
        super().__init__()
        self.sock = sock
        self.email = email

        self.setWindowTitle("Reset Password")
        self.resize(350, 220)

        # ---- Widgets ----
        self.label = QLabel("Enter your new password:")
        self.label.setFont(QFont("Arial", 12))

        self.password_input = QLineEdit()
        self.password_input.setFont(QFont("Arial", 12))
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("New password")

        self.confirm_input = QLineEdit()
        self.confirm_input.setFont(QFont("Arial", 12))
        self.confirm_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm_input.setPlaceholderText("Re-enter password")

        self.reset_button = QPushButton("Reset Password")
        self.reset_button.setFont(QFont("Arial", 12))
        self.reset_button.clicked.connect(self.reset_password)

        self.back_button = QPushButton("Back to Login")
        self.back_button.setFont(QFont("Arial", 12))
        self.back_button.clicked.connect(self.back_to_login)

        # ---- Layout ----
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.confirm_input)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.back_button)

        self.setLayout(layout)

    def reset_password(self):
        password = self.password_input.text().strip()
        confirm = self.confirm_input.text().strip()

        if not password or not confirm:
            QMessageBox.critical(self, "Error", "All fields are required!")
            return

        if password != confirm:
            QMessageBox.critical(self, "Error", "Passwords do not match!")
            return

        if len(password) < 6:
            QMessageBox.critical(self, "Error", "Password must be at least 6 characters!")
            return


        comms.send_with_size({"code": "RESET_PASSWORD", "password": password, "email": self.email})
        print("Resetting password to:", password)

        QMessageBox.information(self, "Success", "Password reset successful!")

        self.close()
        self.login_window = LoginWindow(self.sock)
        self.login_window.show()

    def back_to_login(self):
        self.close()
        self.login_window = LoginWindow(self.sock)
        self.login_window.show()

class EnterCodeWindow(QWidget):
    def __init__(self, sock, email):
        super().__init__()
        self.login_window = LoginWindow(sock)
        self.sock = sock
        self.email = email

        self.setWindowTitle("Enter Reset Code")
        self.resize(350, 200)

        # ---- Widgets ----
        self.label = QLabel("Enter the reset code sent to your email:")
        self.label.setFont(QFont("Arial", 12))

        self.code_input = QLineEdit()
        self.code_input.setFont(QFont("Arial", 12))
        self.code_input.setPlaceholderText("Enter code here")

        self.go_button = QPushButton("Go")
        self.go_button.setFont(QFont("Arial", 12))
        self.go_button.clicked.connect(self.verify_code)

        self.back_button = QPushButton("Back to Login")
        self.back_button.setFont(QFont("Arial", 12))
        self.back_button.clicked.connect(self.back_to_login)

        # ---- Layout ----
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.code_input)
        layout.addWidget(self.go_button)
        layout.addWidget(self.back_button)

        self.setLayout(layout)

    def verify_code(self):
        code = self.code_input.text().strip()

        if not code:
            QMessageBox.critical(self, "Error", "Code cannot be empty!")
            return

        comms.send_with_size({"code": "RESET_CODE", "reset_code": code})
        response = pickle.loads(comms.recv_by_size())

        if response["code"] == "SUCCESS":
            QMessageBox.information(self, "Success", "Code accepted!")
            self.reset_window = ResetPasswordWindow(self.sock, self.email)
            self.reset_window.show()
            self.close()

        else:
            QMessageBox.critical(self, "Error", "Code rejected!")



    def back_to_login(self):
        self.close()
        self.login_window.show()

# ---------------- Run Application ----------------

def main():
    global comms
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 12345))

    app = QApplication(sys.argv)
    entry = EntryWindow(client)
    entry.show()
    sys.exit(app.exec())


def swap_key(client):
    global comms

    key, iv = rsa_key_exchange(client)

    comms = TcpBySize(client, key, iv)


def rsa_key_exchange(client):
    data = pickle.loads(client.recv(1024))
    public_key = serialization.load_pem_public_key(data["public_key"])
    iv = data["iv"]

    key = get_random_bytes(32)


    encrypted_message = public_key.encrypt(
        key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    data = pickle.dumps({"encrypted_key": encrypted_message})

    client.send(data)

    return key, iv

if __name__ == "__main__":
    main()