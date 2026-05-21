import os
import pickle
import socket
import threading
import traceback
import hmac
import time
import random

from Crypto.Random import get_random_bytes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding


from Communication import TcpBySize
from HandleUsers import HandleUsers
from AsyncMessages import AsyncMessages
from Pre_Trained import PreTrainedModel

HOST = '0.0.0.0'
PORT = 12345
ADDRESS = (HOST, PORT)
user_management = HandleUsers()
msg_manager = AsyncMessages()
lock = threading.Lock()
user_by_email = {}
temp_codes = {}
timers = []

keyboard_neighbors = {
    'q': ['w', 'a', 's'],
    'w': ['q', 'e', 'a', 's', 'd'],
    'e': ['w', 'r', 's', 'd', 'f'],
    'r': ['e', 't', 'd', 'f', 'g'],
    't': ['r', 'y', 'f', 'g', 'h'],
    'y': ['t', 'u', 'g', 'h', 'j'],
    'u': ['y', 'i', 'h', 'j', 'k'],
    'i': ['u', 'o', 'j', 'k', 'l'],
    'o': ['i', 'p', 'k', 'l'],
    'p': ['o', 'l'],
    'a': ['q', 'w', 's', 'z', 'x'],
    's': ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c'],
    'd': ['w', 'e', 'r', 's', 'f', 'x', 'c', 'v'],
    'f': ['e', 'r', 't', 'd', 'g', 'c', 'v', 'b'],
    'g': ['r', 't', 'y', 'f', 'h', 'v', 'b', 'n'],
    'h': ['t', 'y', 'u', 'g', 'j', 'b', 'n', 'm'],
    'j': ['y', 'u', 'i', 'h', 'k', 'n', 'm'],
    'k': ['u', 'i', 'o', 'j', 'l', 'm'],
    'l': ['i', 'o', 'p', 'k'],
    'z': ['a', 's', 'x'],
    'x': ['a', 's', 'd', 'z', 'c'],
    'c': ['s', 'd', 'f', 'x', 'v'],
    'v': ['d', 'f', 'g', 'c', 'b'],
    'b': ['f', 'g', 'h', 'v', 'n'],
    'n': ['g', 'h', 'j', 'b', 'm'],
    'm': ['h', 'j', 'k', 'n'],
}
WAITING_TIME = 2
waiting = []
ongoing = []

watch_list = {}

class Game:
    def __init__(self, player1, player2, start):
        self.players = [player1, player2]

        self.player1 = player1
        self.u1 = msg_manager.user_by_sock[self.player1]

        self.player2 = player2

        if type(player2) != PreTrainedModel:
            self.u2 = msg_manager.user_by_sock[self.player2]

        self.start = start

class Client(threading.Thread):
    def __init__(self, sock, addr):
        threading.Thread.__init__(self)
        self.sock = sock
        self.addr = addr

        key, iv = self.swap_key()
        self.comms = TcpBySize(sock, key, iv)
        self.methods = {
            "LOGIN": self.validate_login,
            "SIGNUP": self.validate_signup,
            "SEND_CODE": self.send_reset,
            "RESET_CODE": self.verify_code,
            "RESET_PASSWORD": self.forgot_password,
            "START" : self.start_game,
            "GAME_MSG" : self.add_msg,
            "CHECK" : self.verdict,
            "SCORE_PLACEREQ" : self.get_score,
            "SCOREBOARD_REQ" : self.get_scoreboard,
            "ONGOING_REQ" : self.get_ongoing,
            "GAME_OVER" : self.game_over,
            "WATCH" : self.add_to_watch_list,
            "HISTORY_RES" : self.set_history
        }
        self.model = PreTrainedModel("microsoft/DialoGPT-medium")

        self.logged_in = False
        self.is_turn = False

    def run(self):
        self.sock.settimeout(0.01)

        while True:
            try:
                data: dict = pickle.loads(self.comms.recv_by_size())
            except TimeoutError:
                if self.logged_in:
                    self.send_msgs()
                continue

            if data["code"]:
                print("code:", data["code"])
                to_send = self.methods[data["code"]](data)
            else:
                print("closing socket")
                self.sock.close()
                print(data)
                return

            if to_send:
                self.comms.send_with_size(to_send)

    def game_over(self, data):
        game = [g for g in ongoing if self.sock in g.players]
        if game:
            ongoing.remove(game)

        for spectator in watch_list[game][0]:
            msg_manager.put_msg_by_user({"": {"code": "GAME_OVER"}}, spectator)

    def get_score(self, data):
        scoreboard = user_management.get_scoreboard()

        to_send = pickle.dumps({"code": "SCORE_PLACERES",
                                "score" : scoreboard[msg_manager.user_by_sock[self.sock]],
                                "placement" : list(scoreboard.keys()).index(msg_manager.user_by_sock[self.sock])})

        return to_send

    def get_scoreboard(self, data):
        scoreboard = user_management.get_scoreboard()

        to_send = pickle.dumps({"code" : "SCOREBOARD_RES",
                                "scoreboard" : scoreboard})

        return to_send

    def get_ongoing(self, data):
        num_of_specs = [len(watch_list[game][0]) for game in ongoing]

        return pickle.dumps({"code" : "ONGOING_RES", "ongoing": ongoing, "watching": num_of_specs})

    def add_to_watch_list(self, data):
        game = data["game"]
        p1 = game.player1
        msg_manager.put_msg_by_user({"": {"code": "MSG_HISTORY_REQ"}}, msg_manager.user_by_sock[p1])

        while watch_list[game][1] is None:
            with lock:
                if watch_list[game][1]:
                    watch_list[game][0].append(self.sock)
                    to_send = {"code": "WATCH_START",
                               "HISTORY" : watch_list[game][1]}
                    watch_list[game][1] = None
                    break

        return pickle.dumps(to_send)


    def set_history(self, data):
        game = [g for g in ongoing if g.player1 == self.sock][0]
        watch_list[game][1] = data["HISTORY"]

    def send_msgs(self):
        messages: list[dict] = msg_manager.async_msgs[self.sock]

        if not messages:
            return

        for message in messages:
            data = list(message.values())[0]
            sender = list(message.keys())[0]

            if data["code"] == "CHAT_MSG":
                response = self.create_response("CHAT_MSG", None, {"sender": sender, "msg": data["data"]})
                self.comms.send_with_size(response)
            elif data["code"] == "GAME_OVER":
                response = self.create_response("GAME_OVER", None, None)
                self.comms.send_with_size(response)
            elif data["code"] == "MSG_HISTORY_REQ":
                response = self.create_response("MSG_HISTORY_REQ", None, None)
                self.comms.send_with_size(response)

        msg_manager.async_msgs[self.sock] = []

    def start_game(self, data: dict):
        chatter = "AI"
        print(len(msg_manager.sock_by_user.items()))

        if len(msg_manager.sock_by_user.items()) > 1:
            chatter = "Human" if random.randint(0, 1) == 0 else "AI"

        if chatter == "Human":
            with lock:
                waiting.append(self.sock)
                if len(waiting) == 2:
                    waiting.remove(self.sock)
                    match = waiting.pop(0)

                    t = True if random.random() > 0.5 else False

                    Client.add_game(self.sock, match, t)
                    return pickle.dumps({"code" : "MATCHED", "turn" : t})

            while self.sock in waiting:
                time.sleep(0.001)

            for game in ongoing:
                if self.sock in game.players:
                    return pickle.dumps({"code" : "MATCHED", "turn" : not game.start})


        else:
            time.sleep(random.randint(WAITING_TIME, 3 * WAITING_TIME))
            t = True if random.random() > 0.5 else False
            self.is_turn = t

            with lock:
                Client.add_game(self.sock, self.model, t)

        data = pickle.dumps({"code": "MATCHED", "turn" : self.is_turn})
        self.comms.send_with_size(data)


        if not self.is_turn and chatter == "AI":
            self.send_ai_msg(msg_manager.user_by_sock[self.sock], "Hi")

    @staticmethod
    def add_game(p1, p2, t):
        game = Game(p1, p2, t)
        ongoing.append(game)
        watch_list[game] = ([], None)

    def add_msg(self, data: dict):
        username = data["username"]
        msg = data["msg"]

        to = None
        game = None

        for g in ongoing:
            if g.player1 == self.sock:
                game = g
                if type(g.player2) == PreTrainedModel:
                    self.send_ai_msg(username, msg)
                    return

                else:
                    to = g.u2
            elif g.player2 == self.sock:
                game = g
                to = g.u1

        watching = watch_list[game].append(to)

        for u in watching:
            msg_manager.put_msg_by_user({username: {"code": "CHAT_MSG", "data": msg}}, u)

    def send_ai_msg(self, username, msg):
        text = self.model.generate_response(msg)
        print(text)
        text = Client.spice_up_text(text)
        for i in range(1, len(text)):
            if i % 45 == 0:
                text = text[:i] + "\n  " + text[i:]
        msg_manager.put_msg_by_user({username: {"code": "CHAT_MSG", "data": text}}, username)

    @staticmethod
    def spice_up_text(text: str):
        for i, letter in enumerate(text):
            change = True if random.random() > 0.99 else False
            if change and letter.isalpha():
                options = keyboard_neighbors[letter]
                random.shuffle(options)
                text = text[:i] + options[0] + text[i + 1:]

        capitalized = True if random.randint(0, 150) == 1 else False
        if not capitalized:
            lower = True if random.randint(0, 1) == 0 else False
        else:
            lower = False

        if capitalized:
            return text.upper()
        elif lower:
            return text.lower()
        text = text.replace("'", "")

        if text.endswith("."):
            text = text[:-1]

        return text

    def verdict(self, data: dict):
        verdict = False

        for game in ongoing:
            if self.sock in game.players:
                if data["option"] == "Bot":
                    if type(game.player2) == PreTrainedModel:
                        verdict = True
                elif data["option"] == "Human":
                    if type(game.player1) == socket.socket and type(game.player2) == socket.socket:
                        verdict = True

        if verdict:
            user = [u for u in user_management.users if u.username == msg_manager.user_by_sock[self.sock]][0]

            add_score = 2 if random.random() > 0.5 else 3 if random.random() > 0.7 else 4 if random.random() > 0.5 else 5
            user.score += add_score

            with open("users.pkl", "wb") as f:
                pickle.dump(user_management.users, f)


        return pickle.dumps({"code": "VERDICT", "verdict": verdict})

    def validate_login(self, data: dict):
        username = data["username"]
        password = data["password"]
        email = data["email"]

        with lock:
            if user_management.is_exist(username, password, email):
                msg_manager.add_new_socket(self.sock)
                msg_manager.sock_by_user[username] = self.sock
                msg_manager.user_by_sock[self.sock] = username
                user_by_email[email] = username
                self.logged_in = True
                return self.create_response("LOGINS", None, None)
            return self.create_response("LOGINF", "Login Failed", None)

    def validate_signup(self, data: dict):
        try:
            user_name = data["username"]
            password = data["password"]
            email = data["email"]

            with lock:
                if user_management.is_part_exist(user_name, email):
                    return self.create_response("SIGNUPF", "Sign-up Failed", None)

                user_management.add_user(user_name, password, email)

                with open("users.pkl", "wb") as f:
                    pickle.dump(user_management.users, f)

                return self.create_response("SIGNUPS", None, None)

        except Exception as e:
            print(traceback.format_exc())
            return self.create_response("SIGNUPF", "Sign-up Failed", None)

    def forgot_password(self, data: dict):
        user_management.remove_user(data["email"])
        user_management.add_user(user_by_email[data["email"]], data["password"], data["email"])

    def send_reset(self, data: dict):
        email = data["email"]
        code = random.randint(100000, 1000000)

        temp_codes[email] = code
        timer = threading.Thread(target=self.timer5min, args=email)
        timer.start()
        with lock:
            timers.append(timer)

        # TODO send to email

    @staticmethod
    def timer5min(email):
        time.sleep(300)
        del temp_codes[email]

    def verify_code(self, data: dict):
        sent = temp_codes[data["email"]]
        recv = data["reset_code"]
        match = hmac.compare_digest(str(recv), str(sent))

        if match:
            return self.create_response("SUCCESS", None, None)
        return self.create_response("FAILED", None, None)

    @staticmethod
    def create_response(code, error_description, data):
        response = {
            "code": code,
            "error_description": error_description,
            "data": data
        }
        return pickle.dumps(response)

    def swap_key(self):
        key, iv = self.rsa_key_exchange()

        return key, iv

    def rsa_key_exchange(self):
        private_key, public_key = self.rsa_load_keys()
        iv = get_random_bytes(16)

        pem_public = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        data = {"public_key" : pem_public,
                "iv" : iv}

        self.sock.send(pickle.dumps(data))

        self.sock.settimeout(None)

        encrypted_key = pickle.loads(self.sock.recv(1024))["encrypted_key"]

        try:
            key = private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            return key, iv
        except ValueError:
            return "Decryption failed: Key incorrect"

    @staticmethod
    def rsa_load_keys():
        if not os.path.exists("private_key.pem"):
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,  # 2048 is the industry standard minimum
            )

            pem_private = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.BestAvailableEncryption(b'mypassword')
            )
            with open('private_key.pem', 'wb') as f:
                f.write(pem_private)
        else:
            with open("private_key.pem", "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=b'mypassword',
                    backend=default_backend()
                )

        # 3. Generate and Serialize the Public Key
        if not os.path.exists("public_key.pem"):
            public_key = private_key.public_key()

            pem_public = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            with open("public_key.pem", "wb") as f:
                f.write(pem_public)

        else:
            with open("public_key.pem", "rb") as key_file:
                pem_public = key_file.read()

            public_key = serialization.load_pem_public_key(pem_public)

        return private_key, public_key

    def diffie_hellman(self):
        iv = get_random_bytes(16)

        p = 23
        gen = 5

        private_secret = random.randint(1, 100)

        public_key = pow(gen, private_secret, p)
        data = pickle.dumps({"public_key": public_key,
                             "p": p,
                             "gen": gen,
                             "iv": iv})
        self.sock.send(data)
        public_key2 = int(pickle.loads(self.sock.recv(1024))["public_key"])
        shared_secret = pow(public_key2, private_secret, p)
        key = shared_secret.to_bytes(32, "big")

        return key, iv


def main():
    global user_management
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDRESS)

    try:
        with open("users.pkl", "rb") as f:
            user_management.users = pickle.load(f)

    except Exception as e:
        print(e)

    server.listen(20)
    clients = []
    print("Accepting")

    while True:
        client, addr = server.accept()
        print(addr, "Connected")

        clients.append(client)
        client_thread = Client(client, addr)
        client_thread.start()

    with open("users.pkl", "wb") as f:
        pickle.dump(user_management.users, f)
    [t.join() for t in clients + timers]


if __name__ == '__main__':
    main()