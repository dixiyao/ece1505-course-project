"""The original fedavg, logging down some information."""

from fedavg_client import Client
from fedavg_server import Server
from fedavg_trainer import Trainer

from plato.algorithms.fedavg import Algorithm


def main():
    server = Server(trainer=Trainer, algorithm=Algorithm)
    client = Client(trainer=Trainer, algorithm=Algorithm)
    server.run(client)


if __name__ == "__main__":
    main()
