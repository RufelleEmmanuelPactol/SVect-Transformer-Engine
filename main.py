from multithreaded_server import MultiThreadedServer
import numpy as np
from transformers import BertTokenizer, BertModel
import torch




def main():
    server = MultiThreadedServer("127.0.0.1", 9823)
    server.start()


if __name__ == '__main__':
    main()



