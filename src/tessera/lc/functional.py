"""The orchestrator in LangGraph's Functional API (Phase 5 plan §3.7,
P5-7): the ``orchestration`` switch's ``functional`` value.

The same logic as the ``StateGraph`` in ``tessera.lc.graph``, written as
an ``@entrypoint`` that calls ``@task``s with ordinary Python control flow
— ``if`` statements where the graph has conditional edges, and the
corrective loop as a plain "grade, maybe rewrite and retry once". Both
programming models call the **same** node and step functions
(``tessera.lc.graph``, ``tessera.lc.corrective``) and assemble the run
with the same ``assemble_run``, so they can't drift; the equivalence
tests run both against native.

The rules hold here too: the context arrives per invocation
(``context_schema=GraphContext``; tasks read it with ``get_runtime``),
nothing is bound when the entrypoint is defined, and there is no
checkpointer (no conversation memory).

What the comparison shows (the P5-7 report): the graph declares its
shape — it can be drawn, its node runs are named, and a node can be
streamed or interrupted by name; the Functional API reads as ordinary
code, and its branches and loop are Python, so they are tested as Python.
"""

from __future__ import annotations

from typing import Any

from langgraph.func import entrypoint, task
from langgraph.graph import END
from langgraph.runtime import Runtime, get_runtime

from tessera.lc import corrective as loop
from tessera.lc.graph import (
    GraphContext,
    GraphState,
    LangGraphPipeline,
    after_route,
    assemble_run,
    experts_node,
    generate_b_node,
    generate_node,
    route_node,
    terminal_node,
)
from tessera.pipeline import PipelineRun
from tessera.principal import Principal

FUNCTIONAL_RUN_NAME = "tessera_functional"

# The state keys that append (the graph's reducers), and the rest replace.
APPEND = frozenset({"retrievals", "routing_usage", "generation_usage", "corrective_usage", "retrieval_usage"})


def _apply(state: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if key in APPEND:
            state[key] = [*state.get(key, []), *value]
        else:
            state[key] = value


def _runtime() -> Runtime[GraphContext]:
    return get_runtime(GraphContext)


@task
def route_task(state: GraphState) -> dict[str, Any]:
    return route_node(state, _runtime())


@task
def experts_task(state: GraphState) -> dict[str, Any]:
    return experts_node(state, _runtime())


@task
def generate_b_task(state: GraphState) -> dict[str, Any]:
    return generate_b_node(state, _runtime())


@task
def retrieve_task(state: GraphState) -> dict[str, Any]:
    return loop.retrieve_node(state, _runtime())  # type: ignore[arg-type]


@task
def grade_task(state: GraphState) -> dict[str, Any]:
    return loop.grade_node(state, _runtime())  # type: ignore[arg-type]


@task
def rewrite_task(state: GraphState) -> dict[str, Any]:
    return loop.rewrite_node(state, _runtime())  # type: ignore[arg-type]


@task
def retry_task(state: GraphState) -> dict[str, Any]:
    return loop.retry_node(state, _runtime())  # type: ignore[arg-type]


@task
def generate_task(state: GraphState) -> dict[str, Any]:
    return generate_node(state, _runtime())


@entrypoint(context_schema=GraphContext)
def tessera_flow(inputs: GraphState) -> PipelineRun:
    state: dict[str, Any] = dict(inputs)
    _apply(state, route_task(state).result())
    branch = after_route(state)  # type: ignore[arg-type]
    if branch == "terminal":
        _apply(state, terminal_node(state))  # type: ignore[arg-type]
    elif branch == "experts":
        _apply(state, experts_task(state).result())
        _apply(state, generate_b_task(state).result())
    else:
        # The corrective loop, as code: grade, then at most one rewrite
        # and retry.
        _apply(state, retrieve_task(state).result())
        _apply(state, grade_task(state).result())
        if loop.after_grade(state) != END:  # type: ignore[arg-type]
            _apply(state, rewrite_task(state).result())
            if loop.after_rewrite(state) != END:  # type: ignore[arg-type]
                _apply(state, retry_task(state).result())
        _apply(state, generate_task(state).result())
    return assemble_run(state)  # type: ignore[arg-type]


class FunctionalPipeline(LangGraphPipeline):
    """``LangGraphPipeline``'s parameters, run through the Functional API."""

    def run(self, query: str, principal: Principal | None = None) -> PipelineRun:
        return tessera_flow.invoke(
            self.initial_state(query, principal),
            context=self.context,
            config={"run_name": FUNCTIONAL_RUN_NAME},
        )


__all__ = ["FunctionalPipeline", "tessera_flow"]
