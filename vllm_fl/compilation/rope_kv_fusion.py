# SPDX-License-Identifier: Apache-2.0
"""Bounded functional RoPE + KV update rewrite before vLLM FX splitting."""
import functools
import operator

import torch
from torch import fx
from vllm.logger import init_logger

from vllm_fl.ops import rope_kv_dispatch  # noqa: F401

logger = init_logger(__name__)


def _op(node, name):
    return (isinstance(node, fx.Node) and node.op == "call_function"
            and str(node.target) in (f"vllm.{name}", f"vllm.{name}.default"))


def _view(node):
    if not isinstance(node, fx.Node):
        return False
    return ((node.op == "call_method" and node.target in ("view", "reshape"))
            or (node.op == "call_function" and str(node.target) in (
                "aten.view.default", "aten.reshape.default", "aten._unsafe_view.default")))


def _origin(node):
    while _view(node):
        node = node.args[0]
    return node


def fuse_rope_kv(graph_module):
    graph = graph_module.graph
    count = 0
    for update in list(graph.nodes):
        if not _op(update, "unified_kv_cache_update") or len(update.args) != 3:
            continue
        key, value, layer_name = update.args
        origin = _origin(key)
        if not (isinstance(origin, fx.Node) and origin.op == "call_function"
                and origin.target is operator.getitem and origin.args[1] == 1):
            continue
        rope = origin.args[0]
        if not _op(rope, "fl_rotary_embedding_dispatch") or len(rope.args) != 6:
            continue
        users = list(rope.users)
        if not users or any(u.op != "call_function" or u.target is not operator.getitem
                            or u.args[1] not in (0, 1) for u in users):
            continue
        # One RoPE result must not be attached to two separate cache writers.
        writers = []
        for node in graph.nodes:
            if _op(node, "unified_kv_cache_update"):
                source = _origin(node.args[0])
                if (isinstance(source, fx.Node) and source.op == "call_function"
                        and source.target is operator.getitem
                        and source.args == (rope, 1)):
                    writers.append(node)
        if writers != [update]:
            continue
        order = {n: i for i, n in enumerate(graph.nodes)}
        # Only pure reshape chains may move before the old RoPE. Everything
        # else must already dominate it. No data-dependent work is reordered.
        def movable(node):
            if not isinstance(node, fx.Node) or order[node] < order[rope]:
                return True
            return _view(node) and all(movable(n) for n in node.all_input_nodes)
        if not movable(value) or not movable(layer_name):
            continue
        cloned = {}
        with graph.inserting_before(rope):
            def before(node):
                if order[node] < order[rope]:
                    return node
                if node not in cloned:
                    cloned[node] = graph.node_copy(node, before)
                    cloned[node].meta = dict(node.meta)
                return cloned[node]
            v_before = before(value)
            q, k, cos, sin, positions, interleaved = rope.args
            fused = graph.call_function(torch.ops.vllm.fl_rope_kv_dispatch.default,
                                        (q, k, v_before, cos, sin, positions,
                                         interleaved, layer_name))
            for meta_key in ("example_value", "val"):
                if meta_key in rope.meta and meta_key in update.meta:
                    fused.meta[meta_key] = (*rope.meta[meta_key], update.meta[meta_key])
            outputs = []
            for index in range(3):
                n = graph.call_function(operator.getitem, (fused, index))
                if index == 2:
                    n.meta = dict(update.meta)
                else:
                    for u in users:
                        if u.args[1] == index:
                            n.meta = dict(u.meta)
                            break
                outputs.append(n)
        for user in users:
            user.replace_all_uses_with(outputs[user.args[1]])
            graph.erase_node(user)
        update.replace_all_uses_with(outputs[2])
        graph.erase_node(update)
        graph.erase_node(rope)
        count += 1
    if count:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
        logger.info("LeanScale RoPE+KV graph fusion: replaced=%d", count)
    return count


def install_rope_kv_fusion():
    """Plugin-local hook: vLLM's post-grad passes run after FX splitting."""
    from vllm.platforms import current_platform
    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        return
    from vllm.compilation.backends import VllmBackend
    original = VllmBackend.__call__
    if getattr(original, "_leanscale_rope_kv_fusion", False):
        return

    @functools.wraps(original)
    def with_fusion(self, graph, example_inputs):
        fuse_rope_kv(graph)
        return original(self, graph, example_inputs)

    with_fusion._leanscale_rope_kv_fusion = True
    VllmBackend.__call__ = with_fusion
