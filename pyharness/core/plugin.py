from __future__ import annotations

import asyncio
import contextvars
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable

_MISSING = object()
_DEFAULT_PROFILES: frozenset[str] = frozenset({"enterprise", "standalone", "headless"})


class _EffectEntry:
    """Internal handle tracking a single registered cleanup callback."""

    __slots__ = ("dispose_fn", "disposed")

    def __init__(self, dispose_fn: Callable[[], Any]) -> None:
        self.dispose_fn = dispose_fn
        self.disposed = False


class HarnessPlugin(ABC):
    """
    Cordis-inspired declarative plugin contract for PyHarness runtime extensions.

    Each plugin declares its unique `name` and supported deployment `profiles`
    (`enterprise`, `standalone`, `headless`) and wires services/effects inside `install()`.
    """

    name: str
    profiles: frozenset[str] = _DEFAULT_PROFILES

    @abstractmethod
    async def install(self, ctx: PluginContext) -> None:
        raise NotImplementedError


class PluginContext:
    """
    Cordis-inspired microkernel service container and lifecycle scope manager.

    Responsibilities:
    1. Profile-aware plugin mounting (`mount`) with duplicate protection and atomic rollback
       if `install()` raises partway through.
    2. Scoped service registration (`provide` / `get` / `has`) that automatically binds
       teardown effects to the currently installing plugin scope.
    3. Deterministic LIFO effect unwinding (`effect` / `unmount` / `shutdown`) supporting
       both synchronous and asynchronous disposers.
    """

    def __init__(
        self,
        profile: str = "enterprise",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.profile = profile
        self.config: dict[str, Any] = dict(config or {})
        self._services: dict[str, Any] = {}
        self._mounted: dict[str, HarnessPlugin] = {}
        self._mounting: set[str] = set()
        self._effects: dict[str, list[_EffectEntry]] = {}
        self._root_effects: list[_EffectEntry] = []
        self._active_scope: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"pyharness_plugin_scope_{id(self)}",
            default=None,
        )

    def provide(self, name: str, service: Any) -> None:
        """
        Register a named service on this context.

        When invoked within an active plugin `install()` scope, automatically registers
        a cleanup effect so unmounting that plugin removes the service it provided.
        """
        if not name:
            raise ValueError("Service name must be a non-empty string.")
        self._services[name] = service

        scope = self._active_scope.get()
        if scope is not None:
            def _remove_service() -> None:
                if self._services.get(name) is service:
                    self._services.pop(name, None)

            self.effect(_remove_service)

    def get(self, name: str, default: Any = _MISSING) -> Any:
        """Retrieve a registered service by name, raising `KeyError` if absent and no default."""
        if name in self._services:
            return self._services[name]
        if default is not _MISSING:
            return default
        raise KeyError(f"Service '{name}' is not registered in PluginContext.")

    def has(self, name: str) -> bool:
        """Return True if a service with the given name is currently registered."""
        return name in self._services

    def effect(self, dispose_fn: Callable[[], Any]) -> Callable[[], None]:
        """
        Register a sync or async cleanup callback under the current plugin scope
        (or root scope if called outside `mount()`), returning an idempotent manual
        dispose closure.
        """
        entry = _EffectEntry(dispose_fn)
        scope = self._active_scope.get()
        target_list = self._effects.setdefault(scope, []) if scope is not None else self._root_effects
        target_list.append(entry)

        def _manual_dispose() -> None:
            if entry.disposed:
                return
            entry.disposed = True
            try:
                target_list.remove(entry)
            except ValueError:
                pass
            result = entry.dispose_fn()
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(result)
                else:
                    loop.create_task(result)

        return _manual_dispose

    @staticmethod
    async def _run_effects_lifo(
        entries: list[_EffectEntry],
        *,
        suppress_errors: bool = False,
    ) -> None:
        first_error: Exception | None = None
        for entry in reversed(entries):
            if entry.disposed:
                continue
            entry.disposed = True
            try:
                result = entry.dispose_fn()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                if not suppress_errors and first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    async def mount(self, plugin: HarnessPlugin) -> bool:
        """
        Mount a plugin if its `profiles` include `self.profile` and it is not already mounted.

        If `plugin.install(self)` raises an exception, all partial effects and services
        registered during that install are unwound in LIFO order before re-raising.
        """
        plugin_name = getattr(plugin, "name", "")
        if not plugin_name:
            raise ValueError("HarnessPlugin must define a non-empty 'name'.")
        if self.profile not in plugin.profiles:
            return False
        if plugin_name in self._mounted or plugin_name in self._mounting:
            return False

        self._mounting.add(plugin_name)
        self._effects[plugin_name] = []
        token = self._active_scope.set(plugin_name)
        try:
            await plugin.install(self)
        except BaseException:
            partial_effects = self._effects.pop(plugin_name, [])
            await self._run_effects_lifo(partial_effects, suppress_errors=True)
            raise
        finally:
            self._active_scope.reset(token)
            self._mounting.discard(plugin_name)

        self._mounted[plugin_name] = plugin
        return True

    async def unmount(self, plugin_name: str) -> bool:
        """
        Unmount a previously mounted plugin, executing all of its registered effects
        in strict LIFO reverse order (supporting both sync and async disposers).
        """
        if plugin_name not in self._mounted:
            return False
        self._mounted.pop(plugin_name, None)
        entries = self._effects.pop(plugin_name, [])
        await self._run_effects_lifo(entries, suppress_errors=False)
        return True

    async def shutdown(self) -> None:
        """Unmount all mounted plugins in reverse mount order and run any root effects."""
        first_error: Exception | None = None
        for plugin_name in list(reversed(list(self._mounted.keys()))):
            try:
                await self.unmount(plugin_name)
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if self._root_effects:
            root_entries = self._root_effects
            self._root_effects = []
            try:
                await self._run_effects_lifo(root_entries, suppress_errors=False)
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if first_error is not None:
            raise first_error

    def mounted_plugins(self) -> list[str]:
        """Return currently mounted plugin names in mount order."""
        return list(self._mounted.keys())
