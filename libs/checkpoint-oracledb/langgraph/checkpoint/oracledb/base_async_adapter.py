import asyncio

from .base import OracleDBSaver


class OracleDBSaverAsyncAdapter(OracleDBSaver):
    """Async wrapper around sync OracleDBSaver for conformance testing."""

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(self, config, **kw):
        items = await asyncio.to_thread(lambda: list(self.list(config, **kw)))
        for item in items:
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(
            self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await asyncio.to_thread(
            self.put_writes, config, writes, task_id, task_path
        )

    async def adelete_thread(self, thread_id):
        return await asyncio.to_thread(self.delete_thread, thread_id)

    async def adelete_for_runs(self, run_ids):
        return await asyncio.to_thread(self.delete_for_runs, run_ids)

    async def acopy_thread(self, source_thread_id, target_thread_id):
        return await asyncio.to_thread(
            self.copy_thread, source_thread_id, target_thread_id
        )

    async def aprune(self, thread_ids, *, strategy="keep_latest"):
        return await asyncio.to_thread(self.prune, thread_ids, strategy=strategy)
