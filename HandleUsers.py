import secrets
import string
import hashlib
import pickle
import os


class User:
    def __init__(self, username, password, email):
        self.username = username
        self.password = password
        self.email = email
        self.score = 0


class HandleUsers:
    def __init__(self):
        self.users = []
        if not os.path.exists("pepper.pkl"):
            with open("pepper.pkl", "wb") as f:
                pickle.dump(random_string(), f)

    def get_scoreboard(self):
        scores = {u: s for u, s in zip([u.username for u in self.users], [s.score for s in self.users])}
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def is_exist(self, username, password, email):
        with open("pepper.pkl", "rb") as f:
            pepper = pickle.load(f)

        if username not in [u.username for u in self.users]:
            return False

        user = [u for u in self.users if u.username == username][0]

        stored_hash, salt = user.password
        computed_hash = hashlib.sha256((password + salt + pepper).encode()).hexdigest()

        if stored_hash != computed_hash:
            return False

        if user.email != email:
            return False

        return True

    def is_online(self, username):
        return username in [u.username for u in self.users]

    def is_part_exist(self, username, email):
        if username in [u.username for u in self.users]:
            return True
        if email in [u.email for u in self.users]:
            return True
        return False

    def add_user(self, username, password, email):
        salt = random_string()
        with (open("pepper.pkl", "rb")) as f:
            pepper = pickle.load(f)

        self.users.append(User(username, [hashlib.sha256((password + salt + pepper).encode()).hexdigest(), salt], email))

    def remove_user(self, username):
        with open("users.pkl", "rb") as f:
            data = pickle.load(f)

        user = [u for u in self.users if u.username == username][0]

        self.users.remove(user)
        data.remove(user)

        with open("users.pkl", "wb") as f:
            pickle.dump(data, f)

def random_string(length=10):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))