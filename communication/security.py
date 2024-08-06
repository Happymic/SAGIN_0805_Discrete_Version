from cryptography.fernet import Fernet

# In a real-world scenario, you'd want to manage this key more securely
key = Fernet.generate_key()
fernet = Fernet(key)

def encrypt_message(message):
    return fernet.encrypt(message.encode())

def decrypt_message(encrypted_message):
    return fernet.decrypt(encrypted_message).decode()