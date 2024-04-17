"""Class to manage communication between different worlds."""
from __future__ import annotations

import asyncio
import logging
import threading
from asyncio import Queue as AsyncQ
from queue import Empty
from queue import SimpleQueue as SimpleSyncQ
from typing import TYPE_CHECKING, Union

import torch.distributed as dist
from torch import Tensor
from torch.distributed import Work

from multiworld.threadsafe_async import run_async

if TYPE_CHECKING:
    from torch.distributed.world_manager import WorldManager

logger = logging.getLogger(__name__)


WAIT_TIME_FOR_FIFO = 0.2  # 200 ms


_errors_to_handle = [
    "Connection closed by peer",
    "Connection reset by peer",
    "NCCL communicator was aborted",
]


class BrokenWorldException(Exception):
    """Raise this exception when world is broken."""

    def __init__(self, world_name: str):
        """Initialize exception instance."""
        self._world_name = world_name

    def __str__(self):
        """Return exception string."""
        return f"broken world: {self._world_name}"

    pass


class WorkStatus:  # noqa: D101
    SUCCESS = 0
    BROKEN = 1  # To indicate if the world is broken


class CommunicationType:  # noqa: D101
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
        self.status: WorkStatus = WorkStatus.SUCCESS


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

    def __del__(self):
        """Cleanup the class instance."""
        for world_name in self._communication_threads:
            self.remove_world(world_name)

    def add_world(self, world_name):
        """Add a new world to the world comm manager."""
        input_q = SimpleSyncQ()
        event_q = AsyncQ()
        self._communication_commands[world_name] = (input_q, event_q)

        # NOTE(pranav): Might want to create separate threads for sending and receiving
        self._communication_threads[world_name] = threading.Thread(
            target=self._communication_thread,
            args=(world_name,),
            daemon=True,
        )
        self._communication_threads[world_name].start()

    def remove_world(self, world_name):
        """Remove a world from the world comm manager."""
        logger.debug(f"remove world {world_name}")
        try:
            # delete command for world as it is no longer needed
            del self._communication_commands[world_name]
        except KeyError:
            pass

        try:
            # delete thread entry for world as it is no longer needed
            del self._communication_threads[world_name]
        except KeyError:
            pass

    def _handle_work(self, work: Work) -> None:
        while True:
            try:
                # Enable lazy thread termination by using timeout and event
                _ = work.wait()
                # if no exception and we reach here, the work is done;
                # then we get out of the while loop
                break
            except RuntimeError as e:
                err_msg = str(e)
                logger.debug(err_msg)
                for error_snippet in _errors_to_handle:
                    if error_snippet in err_msg:
                        return WorkStatus.BROKEN

                raise e

        return WorkStatus.SUCCESS

    def _communication_thread(self, world_name: str):
        """Thread function to manage communication between worlds."""
        logger.debug(f"starting communication thread for {world_name}")
        input_q = self._communication_commands[world_name][0]
        event_q = self._communication_commands[world_name][1]

        while True:
            # This call blocks indefinitely until a command is received
            works: list[Work] = input_q.get()
            # comm_obj: CommObject = input_q.get()

            status = WorkStatus.SUCCESS
            for work in works:
                status = self._handle_work(work)
                if status != WorkStatus.SUCCESS:
                    break

            _, _ = run_async(event_q.put(status), self._loop)

            if status == WorkStatus.BROKEN:
                logger.debug(f"world {world_name} is broken")
                break

        try:
            # delete command for world as it is no longer needed
            del self._communication_commands[world_name]
        except KeyError:
            pass

        try:
            # delete thread entry for world as it is no longer needed
            del self._communication_threads[world_name]
        except KeyError:
            pass
        logger.debug(f"terminating comm. thread for {world_name}")

    async def send(
        self, tensors: Union[Tensor, list[Tensor]], world_name: str, rank: int
    ) -> None:
        """Send a tensor or a list of tensors to a specific rank in a world.

        This method supports batched send.
        """
        self._world_manager.set_world(world_name)

        self._send(tensors, world_name, rank)

        event_q = self._communication_commands[world_name][1]
        status = await event_q.get()
        if status == WorkStatus.BROKEN:
            self._world_manager.remove_world(world_name)
            raise BrokenWorldException(f"{world_name}")

    def _send(
        self, tensors: Union[Tensor, list[Tensor]], world_name: str, rank: int
    ) -> None:
        # Catch any errors due to worker failures
        try:
            if isinstance(tensors, Tensor):
                tensors = list(tensors)

            works = list()
            for tensor in tensors:
                work = dist.isend(tensor, dst=rank)
                works.append(work)

            input_q = self._communication_commands[world_name][0]
            input_q.put(works)

        except RuntimeError as e:
            self._handle_error(e, world_name)

    async def recv(
        self, tensors: Union[Tensor, list[Tensor]], world_name: str, rank: int
    ) -> None:
        """Receive tensor(s) from a specific rank in a world.

        This method supports batched receive.
        """
        self._world_manager.set_world(world_name)

        self._recv(tensors, world_name, rank)

        event_q = self._communication_commands[world_name][1]
        status = await event_q.get()
        if status == WorkStatus.BROKEN:
            self._world_manager.remove_world(world_name)
            raise BrokenWorldException(f"{world_name}")

    def _recv(self, tensors: Union[Tensor, list[Tensor]], world_name: str, rank: int):
        # Catch any errors due to worker failures
        try:
            if isinstance(tensors, Tensor):
                tensors = list(tensors)

            works = list()
            for tensor in tensors:
                work = dist.irecv(tensor, src=rank)
                works.append(work)

            input_q = self._communication_commands[world_name][0]
            input_q.put(works)

        except RuntimeError as e:
            self._handle_error(e, world_name)

    async def recv_fifo(
        self, tensor: Tensor, senders: list[tuple[str, int]]
    ) -> tuple[Tensor, str, int]:
        """Receive tensor from a list of senders in a fifo fashion.

        This method will be deprecated.
        """
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
            while True:
                try:
                    comm_obj: CommObject = self._tensor_rx_q.get(
                        timeout=WAIT_TIME_FOR_FIFO
                    )
                    break
                except Empty:
                    # in case of timeout, yield control back to the event loop
                    # so that the event loop can schedule some pending tasks
                    await asyncio.sleep(0)
                    continue

            if comm_obj.status == WorkStatus.BROKEN:
                raise BrokenWorldException(f"{comm_obj.world_name}")
            else:
                yield (comm_obj.tensor, comm_obj.world_name, comm_obj.rank)

    def _handle_error(self, error: RuntimeError, world_name: str):
        error_message = str(error)

        for error_snippet in _errors_to_handle:
            if error_snippet in error_message:
                logger.debug(f"broken world: {error_message}")
                self._world_manager.remove_world(world_name)
                raise BrokenWorldException(f"{world_name}")

        raise error

    @property
    def rx_q(self):
        """Return the rx queue for received tensors."""
        return self._tensor_rx_q
