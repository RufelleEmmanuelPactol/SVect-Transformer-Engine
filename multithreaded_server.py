import multiprocessing
import socket
import json
import threading
from module_loader import load_module

def decode_worker(args):
    vector_space, transformer, key, encoded_byte_array = args
    module = load_module(vector_space, transformer)
    if module is not None:
        if isinstance(encoded_byte_array, list):
            # Conversion from signed to unsigned byte array is handled here
            byte_array = bytes([(value) % 256 for value in encoded_byte_array])
        else:
            byte_array = encoded_byte_array

        result = {
            key: module.transform(byte_array).tolist()  # Pass byte_array directly
        }
        return result








class MultiThreadedServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}")

    def start(self):
        while True:
            client_socket, client_address = self.server_socket.accept()
            print(f"Accepted connection from {client_address}")
            client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
            client_thread.start()

    def handle_client(self, client_socket):
        try:
            data_length_bytes = client_socket.recv(4)
            data_length = int.from_bytes(data_length_bytes, byteorder='big')

            received_data = b''
            while len(received_data) < data_length:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                received_data += chunk

            if not received_data:
                return

            decoded_data = received_data.decode('utf-16')
            print(decoded_data)# Assuming data is UTF-16 encoded
            json_data = json.loads(decoded_data)

            vector_space = json_data.get('vector-space')
            transformer_name = json_data.get('transformer-name')
            raw_transform = json_data.get('raw-transform')

            # Use multiprocessing pool for decoding
            pool = multiprocessing.Pool()

            # Create a list of arguments for the decode_worker function
            decode_args = [(vector_space, transformer_name, key, encoded_byte_array) for key, encoded_byte_array in
                           raw_transform.items()]

            # Map the decode_worker function to the list of arguments using the pool
            results = pool.map(decode_worker, decode_args)

            pool.close()
            pool.join()

            # Combine results into a single dictionary
            response_dict = {}
            for result in results:
                response_dict.update(result)

            response_data = {'response': response_dict}
            response_data = json.dumps(response_data)
            response_bytes = response_data.encode('utf-16')
            response_length_bytes = len(response_bytes).to_bytes(4, byteorder='big')

            client_socket.sendall(response_length_bytes + response_bytes)

        except Exception as e:
            raise e
        finally:
            client_socket.close()


