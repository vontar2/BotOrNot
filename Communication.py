__author__ = 'Yossi'
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# from  tcp_by_size import send_with_size ,recv_by_size

class TcpBySize:
    def __init__(self, sock, key, iv):
        self.SIZE_HEADER_FORMAT = "0000000|" # n digits for data size + one delimiter
        self.size_header_size = len(self.SIZE_HEADER_FORMAT)
        self.TCP_DEBUG = True
        self.LEN_TO_PRINT = 100
        self.sock = sock
        self.key = key
        self.iv = iv

    def encrypt_cbc(self, bdata):
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
        ciphertext = cipher.encrypt(pad(bdata, AES.block_size))

        return ciphertext

    def decrypt_cbc(self, ciphertext):
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
        decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)

        return decrypted

    def recv_by_size(self):
        size_header = b''
        data_len = 0
        while len(size_header) < self.size_header_size:
            _s = self.sock.recv(self.size_header_size - len(size_header))
            if _s == b'':
                size_header = b''
                break
            size_header += _s
        data = b''
        if size_header != b'':
            data_len = int(size_header[:self.size_header_size - 1])
            while len(data) < data_len:
                _d = self.sock.recv(data_len - len(data))
                if _d == b'':
                    data = b''
                    break
                data += _d

        if self.TCP_DEBUG and size_header != b'':
            print("\nRecv(%s)>>>" % (size_header,), end='')
            print("%s" % (data[:min(len(data), self.LEN_TO_PRINT)],))
        if data_len != len(data):
            data = b''  # Partial data is like no data !
        return self.decrypt_cbc(data)

    def send_with_size(self, bdata):
        if type(bdata) == str:
            bdata = bdata.encode()

        bdata = self.encrypt_cbc(bdata)

        len_data = len(bdata)
        header_data = str(len_data).zfill(self.size_header_size - 1) + "|"

        bytea = bytearray(header_data,encoding='utf8') + bdata

        self.sock.send(bytea)
        if self.TCP_DEBUG and  len_data > 0:
            print ("\nSent(%s)>>>" % (len_data,), end='')
            print ("%s"%(bytea[:min(len(bytea),self.LEN_TO_PRINT)],))
