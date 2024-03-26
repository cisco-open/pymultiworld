"""Class to manage communication between different worlds."""
from __future__ import annotations

import asyncio
import logging
import threading
from asyncio import Queue as AsyncQ
from queue import SimpleQueue as SimpleSyncQ
from typing import TYPE_CHECKING, Union

import torch.distributed as dist
from torch import Tensor
from torch.distributed import Work

from multiworld.threadsafe_async import run_async

if TYPE_CHECKING:
    from torch.distributed.world_manager import WorldManager

logger = logging.getLogger(__name__)


_errors_to_handle = [
    "Connection closed by peer",
    "Connection reset by peer",
    "NCCL communicator was aborted",
]


class CommunicationType:
    SEND = 1
    RECV = 2
    RECV_FIFO = 3


class CommObject:
    """Internal communication ojbect."""

    def __init__(
        self,
        command: CommunicationType,
        work: Work,
        tensor: Union[Tensor, None],
        world_name: Union[str, None],
        rank: Union[int, None],
    ):
        """Initialize an instance."""
        self.command = command
        self.work = work
        self.tensor = tensor
        self.world_name = world_name
        self.rank = rank


class WorldCommunicator:
    """
    Class to manage communication between different worlds.

    NOTE: If using WorldCommunicationManager, use the API provided
    by the WorldCommunicator to create and manage worlds along with
    their communication links. Do not use the WorldManager API directly.
    """

    def __init__(self, world_manager: WorldManager):
        """Initialize a class instance."""
        self._world_manager = world_manager
        self._communication_threads = {}
        self._communication_commands = {}

        self._tensor_rx_q = SimpleSyncQ()

        self._loop = asyncio.get_running_loop()

    def add_world(self, world_name):
        """Add a new world to the world comm manager."""
        input_q = SimpleSyncQ()
        event_q = AsyncQ()
        self._communication_commands[world_name] = (input_q, event_q)

        # NOTE(pranav): Might want to create separate threads for sending and receiving
        self._communication_threads[world_name] = threading.Thread(
            target=self._communication_thread, args=(world_name,)
        )
        self._communication_threads[world_name].start()

    def remove_world(self, world_name):
        """Remove a world from the world comm manager."""
        # TODO: stop the thread associated with the world to remove
        del self._communication_commands[world_name][0]
        del self._communication_commands[world_name][1]
        del self._communication_threads[world_name]

    def _communication_thread(self, world_name: str):
        """Thread function to manage communication between worlds."""
        logger.debug(f"starting communication thread for {world_name}")

        while True:
            input_q = self._communication_commands[world_name][0]
            # This call blocks indefinitely until a command is received
            comm_obj: CommObject = input_q.get()
            res = comm_obj.work.wait()

            if comm_obj.command == CommunicationType.RECV_FIFO:
                self._tensor_rx_q.put(comm_obj)
            else:
                event_q = self._communication_commands[world_name][1]
                _, _ = run_async(event_q.put(res), self._loop)

    async def send(self, tensor: Tensor, world_name: str, rank: int) -> None:
        """Send a tensor to a specific rank in a world."""
        self._world_manager.set_world(world_name)

        # Catch any errors due to worker failures
        try:
            work = dist.isend(tensor, dst=rank)

            input_q = self._communication_commands[world_name][0]
            comm_obj = CommObject(CommunicationType.SEND, work, None, None, None)
            input_q.put(comm_obj)

        except RuntimeError as e:
            self._handle_error(e, world_name)

        event_q = self._communication_commands[world_name][1]
        _ = await event_q.get()

    async def recv(self, tensor: Tensor, world_name: str, rank: int) -> None:
        """Receive a tensor from a specific rank in a world."""
        self._world_manager.set_world(world_name)

        # Catch any errors due to worker failures
        try:
            work = dist.irecv(tensor, src=rank)

            input_q = self._communication_commands[world_name][0]
            comm_obj = CommObject(CommunicationType.RECV, work, None, None, None)
            input_q.put(comm_obj)

        except RuntimeError as e:
            self._handle_error(e, world_name)

        event_q = self._communication_commands[world_name][1]
        _ = await event_q.get()

    def recv_fifo(
        self, tensor: Tensor, senders: list[tuple[str, int]]
    ) -> tuple[Tensor, str, int]:
        """Receive tensor from a list of senders in a fifo fashion."""
        fail_count = 0
        for world_name, rank in senders:
            buffer = tensor.detach().clone()

            self._world_manager.set_world(world_name)

            # Catch any errors due to worker failures
            try:
                work = dist.irecv(buffer, src=rank)

                input_q = self._communication_commands[world_name][0]
                comm_obj = CommObject(
                    CommunicationType.RECV_FIFO, work, buffer, world_name, rank
                )
                input_q.put(comm_obj)

            except RuntimeError as e:
                self._handle_error(e, world_name)
                fail_count += 1

        count = len(senders)
        while count > fail_count:
            count -= 1
            comm_obj: CommObject = self._tensor_rx_q.get()
            yield (comm_obj.tensor, comm_obj.world_name, comm_obj.rank)

    def _handle_error(self, error: RuntimeError, world_name: str):
        error_message = str(error)

        for error_snippet in _errors_to_handle:
            if error_snippet in error_message:
                logger.warn(f"Ignoring error: {error_message}")
                self.remove_world(world_name)
                break
        else:
            raise error

    @property
    def rx_q(self):
        """Return the rx queue for received tensors."""
        return self._tensor_rx_q
