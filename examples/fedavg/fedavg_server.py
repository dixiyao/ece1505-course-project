"""Need to log down G_i"""
import os
import logging
import random
import numpy as np
import pandas as pd
import time
import heapq

from plato.utils import csv_processor

from plato.config import Config
from plato.servers import fedavg
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import all_inclusive


class Server(fedavg.Server):
    """Log some necessary information."""

    def configure(self) -> None:
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        super().configure()

        total_rounds = Config().trainer.rounds
        target_accuracy = None
        target_perplexity = None

        if hasattr(Config().trainer, "target_accuracy"):
            target_accuracy = Config().trainer.target_accuracy
        elif hasattr(Config().trainer, "target_perplexity"):
            target_perplexity = Config().trainer.target_perplexity

        if target_accuracy:
            logging.info(
                "Training: %s rounds or accuracy above %.1f%%\n",
                total_rounds,
                100 * target_accuracy,
            )
        elif target_perplexity:
            logging.info(
                "Training: %s rounds or perplexity below %.1f\n",
                total_rounds,
                target_perplexity,
            )
        else:
            logging.info("Training: %s rounds\n", total_rounds)

        self.init_trainer()

        # Prepares this server for processors that processes outbound and inbound
        # data payloads
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Server", server_id=os.getpid(), trainer=self.trainer
        )

        if not (hasattr(Config().server, "do_test") and not Config().server.do_test):
            if self.datasource is None and self.custom_datasource is None:
                self.datasource = datasources_registry.get(client_id=0)
            elif self.datasource is None and self.custom_datasource is not None:
                self.datasource = self.custom_datasource()

            self.testset = self.datasource.get_test_set()
            if hasattr(Config().data, "testset_size"):
                self.testset_sampler = all_inclusive.Sampler(
                    self.datasource, testing=True
                )

        # Initialize the test accuracy csv file if clients compute locally
        accuracy_csv_file = (
            f"{Config().params['result_path']}/{os.getpid()}_accuracy.csv"
        )
        accuracy_headers = [
            "round",
            "client_id",
            "gbound",
            "round_time",
            "samples",
        ]
        csv_processor.initialize_csv(
            accuracy_csv_file, accuracy_headers, Config().params["result_path"]
        )

    def load_probability(self):
        """Load the probability"""
        prob = (
                np.zeros(Config().clients.total_clients)
            )
        # uniform
        if Config().clients.sample == "uniform":

            for i in range(100):
                prob[i]=0.01
        if Config().clients.sample == "pi":
            prob_f=pd.read_csv('/home/dixi/plato-FedNAS/examples/fedavg/fedavg_p.csv')
            for i in range(100):
                prob[i]=prob_f['p'][i]
        return prob
    
    def choose_clients(self, clients_pool, clients_count):
        assert clients_count <= len(clients_pool)
        random.setstate(self.prng_state)

        prob = self.load_probability()
        # Select clients randomly
        selected_clients = np.random.choice(
            np.array(clients_pool), clients_count, p=prob
        ).tolist()
        # random.sample(clients_pool, clients_count)

        self.prng_state = random.getstate()
        logging.info("[%s] Selected clients: %s", self, selected_clients)
        return selected_clients
        
    async def process_client_info(self, client_id, sid):
        """Processes the received metadata information from a reporting client."""
        # First pass through the inbound_processor(s), if any
        self.client_payload[sid] = self.inbound_processor.process(
            self.client_payload[sid]
        )

        if self.comm_simulation:
            if (
                hasattr(Config().clients, "compute_comm_time")
                and Config().clients.compute_comm_time
            ):
                self.reports[sid].comm_time = (
                    self.downlink_comm_time[client_id]
                    + self.uplink_comm_time[client_id]
                )
            else:
                self.reports[sid].comm_time = 0
        else:
            self.reports[sid].comm_time = time.time() - self.reports[sid].comm_time

        if hasattr(self.reports[sid], "client_id"):
            # When the client is responding to an urgent request for an update, it will
            # store its client ID in its report
            client_id = self.reports[sid].client_id

        try:
            start_time = self.training_clients[client_id]["start_time"]
        except:
            start_time = time.time()
        finish_time = (
            self.reports[sid].training_time + self.reports[sid].comm_time + start_time
        )
        
        try:
            starting_round = self.training_clients[client_id]["starting_round"]
        except:
            starting_round = 1

        if Config().is_central_server():
            self.comm_overhead += self.reports[sid].edge_server_comm_overhead

        client_info = (
            finish_time,  # sorted by the client's finish time
            client_id,  # in case two or more clients have the same finish time
            {
                "client_id": client_id,
                "sid": sid,
                "starting_round": starting_round,
                "start_time": start_time,
                "report": self.reports[sid],
                "payload": self.client_payload[sid],
            },
        )

        heapq.heappush(self.reported_clients, client_info)
        self.current_reported_clients[client_info[2]["client_id"]] = True
        try:
            del self.training_clients[client_id]
        except:
            pass

        if self.asynchronous_mode and self.simulate_wall_time:
            self.training_sids.remove(client_info[2]["sid"])

        await self._process_clients(client_info)
   
