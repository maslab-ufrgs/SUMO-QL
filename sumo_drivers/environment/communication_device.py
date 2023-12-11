from __future__ import annotations

import random as rd
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from sumolib.net.node import Node

if TYPE_CHECKING:
    from sumo_drivers.environment.sumo_environment import SumoEnvironment


class CommunicationDevice:
    """Class that is responsible for taking care of the infrastructure information such as the expected rewards for
    the neighboring links.

    Args:
        node (sumolib.net.node): Node that the CommunicationDevice is installed
        max_queue_size (int): Max number of informed rewards that can be stored in the commDev
        comm_success_rate (float): Value between 0 and 1 that stores the success rate of the communication
        environment (SumoEnvironment): Stores the object of the environment the commDev is in
    """

    def __init__(
        self,
        node: Node,
        max_queue_size: int,
        comm_success_rate: float,
        environment: SumoEnvironment,
        od_graph: bool,
    ) -> None:
        self.__node = node
        self.__max_queue_size = max_queue_size
        self.__comm_success_rate = comm_success_rate
        self.__environment = environment
        self.__data: dict[str, list[tuple[np.ndarray, str]]] = {
            link.getID(): list() for link in node.getIncoming()
        }
        self.__od_graph: bool = od_graph

        rd.seed(datetime.now().timestamp())

    @property
    def communication_success(self) -> bool:
        """Property that returns if the communication should fail or succeed based on a random number taken and the
        success rate stored.

        Returns:
            boolean: Information regarding whether the communication attempt should fail or succeed
        """
        return rd.random() <= self.__comm_success_rate

    def update_stored_rewards(
        self, od_pair: str, link: str, reward: np.ndarray
    ) -> None:
        """This method receives a link and a reward, so it stores the reward communicated for the given link. If the
        queue is already full the oldest reward information is discarded to make room for the newest one to be inserted.
           If the link is not part of the commDev's neighboring links, the method raises a RunTimeError.

        Args:
            link (str): String with link ID to update data
            reward (np.ndarray): Reward values to insert in link queue
        """
        if link not in self.__data.keys():
            raise RuntimeError("Link is not connected to commDev's node")

        if len(self.__data[link]) == self.__max_queue_size:
            self.__data[link].pop(0)
        self.__data[link].append((reward, od_pair))

    def get_expected_reward(self, od_pair: str, link: str) -> np.ndarray:
        """This method returns the expected reward for the given link. If there's no reward stored for the given link,
        the method returns 0. It returns a list with the averages of the stored rewards otherwise.
           If the link is not part of the commDev's neighboring links, the method raises a RunTimeError.

        Args:
            link (str): String with the link ID to get the expected reward

        Returns:
            np.ndarray: List of averages of the stored rewards for the given link if there is data stored or 0.0 otherwise
        """
        nobj = len(self.__environment.objectives.known_objectives)

        if link not in self.__data.keys():
            raise RuntimeError("Link is not connected to commDev's node")

        link_data = [
            reward
            for reward, od in self.__data[link]
            if self._are_neighbors(od_pair, od)
        ]
        if len(link_data) > 0:
            data_mean: np.ndarray = np.mean(
                link_data,
                axis=0,
            )
            if data_mean.size == nobj:  # if the amount of data is correct
                return data_mean
            else:
                print(f"Size data mean doesn't match number of objectives for {link}")
        # else:
        #     print(f"Empty link data list for {link}")

        return np.zeros(shape=nobj)

    def _are_neighbors(self, current_od: str, rw_od: str) -> bool:
        if not self.__od_graph:
            return True
        if current_od == rw_od:
            return True

        graph_neighbors = self.__environment.get_graph_neighbors()
        neighbors = self.get_graph_neighbors_interval(
            graph_neighbors[current_od], self.__environment.current_step
        )
        return rw_od in neighbors

    def get_graph_neighbors_interval(
        self, graph_neighbors: dict, current_step: int
    ) -> list:
        number_of_intervals = len(graph_neighbors)
        i = 0
        for interval in graph_neighbors:
            if i == number_of_intervals - 1:
                if interval[0] <= current_step <= interval[1]:
                    return graph_neighbors[interval]
            else:
                if interval[0] < current_step <= interval[1]:
                    return graph_neighbors[interval]
            i += 1
        # print("Interval not found, returning empty list")
        return []

    def get_outgoing_links_expected_rewards(
        self, od_pair: str
    ) -> dict[str, np.ndarray]:
        """Returns a dictionary containing the expected rewards from all the outgoing links from the commDev's node.

        Returns:
            dict[str, float]: dictionary containing expected rewards from all outgoing links from the commDev's node.
            Being that the keys are the links ID and the values are the expected rewards.
        """
        if not self.__od_graph:
            return self._link_data_wo_od_graph()

        return self._link_data_od_graph(od_pair)

    def _link_data_od_graph(self, od_pair: str) -> dict[str, np.ndarray]:
        links_data = dict()

        for link in self.__node.getOutgoing():
            link_id = link.getID()

            # gets data of neighboring commdev
            neighboring_comm_dev = self.__environment.get_comm_dev(
                link.getToNode().getID()
            )
            links_data[link_id] = neighboring_comm_dev.get_expected_reward(
                od_pair, link_id
            )

        return links_data

    def _link_data_wo_od_graph(self) -> dict[str, np.ndarray]:
        links_data = dict()
        for link in self.__node.getOutgoing():
            link_id = link.getID()

            # gets data of neighboring commdev
            neighboring_comm_dev = self.__environment.get_comm_dev(
                link.getToNode().getID()
            )
            links_data[link_id] = neighboring_comm_dev.get_expected_reward("", link_id)

            # gets data of commdev of graph neighbor link
            graph_neighbors = self.__environment.get_graph_neighbors()
            if link_id in list(graph_neighbors.keys()):
                current_step = self.__environment.current_step
                graph_neighbors_link_interval = self.get_graph_neighbors_interval(
                    graph_neighbors[link_id], current_step
                )

                for link_graph_neighbor in graph_neighbors_link_interval:
                    node_id = self.__environment.get_link_destination(
                        link_graph_neighbor
                    )
                    graph_comm_dev = self.__environment.get_comm_dev(node_id)

                    links_data[
                        link_graph_neighbor
                    ] = graph_comm_dev.get_expected_reward("", link_graph_neighbor)

        return links_data
