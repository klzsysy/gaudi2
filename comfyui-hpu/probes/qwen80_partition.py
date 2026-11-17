"""Process-local exact bitset representation of FX partition reachability."""
import inspect
import textwrap

import torch.fx.passes.infra.partitioner as upstream
from habana_frameworks.torch.dynamo.compile_backend import partitioner, passes


class Dependencies:
    def __init__(self, graph):
        self.nodes = tuple(graph.graph.nodes)
        self.bits = {node: 1 << index for index, node in enumerate(self.nodes)}
        self.downstreams = {}
        for node in reversed(self.nodes):
            mask = 0
            for user in node.users:
                mask |= self.bits[user] | self.downstreams[user]
            self.downstreams[node] = mask


source = textwrap.dedent(inspect.getsource(upstream.CapabilityBasedPartitioner.propose_partitions))
source = source.replace('    new_partition_id = itertools.count()',
                        '    new_partition_id = itertools.count()\n    partition_bits = {}')
start = source.index('    def downstream_partitions(')
end = source.index('    # try to merge partition', start)
source = source[:start] + '''    def downstream_mask(user_nodes):
        mask = 0
        for node in user_nodes:
            mask |= self.dependency_viewer.bits[node] | self.dependency_viewer.downstreams[node]
        return mask

    def downstream_partitions(user_nodes, source_ids):
        mask = downstream_mask(user_nodes)
        targets = set()
        for partition_id, bits in partition_bits.items():
            if mask & bits:
                targets.add(partition_id)
                targets.update(partition_map[partition_id])
        targets.difference_update(source_ids)
        return targets

''' + source[end:]
start = source.index('        def dfs_iter_find_cycle(')
end = source.index('        # find new partition users', start)
source = source[:start] + '''        def dfs_iter_find_cycle(all_user_nodes):
            mask = downstream_mask(all_user_nodes)
            if mask & (partition_bits[self_id] | partition_bits[other_id]):
                return True
            for partition_id, bits in partition_bits.items():
                if mask & bits:
                    reachable = partition_map[partition_id]
                    if self_id in reachable or other_id in reachable:
                        return True
            return False

''' + source[end:]
replacements = {
    '        del partitions_by_id[removed_id]':
        '        del partitions_by_id[removed_id]\n        partition_bits[merge_id] |= partition_bits.pop(removed_id)',
    '            partitions_by_id[assignment[node]].remove_node(node)':
        '            partitions_by_id[assignment[node]].remove_node(node)\n            partition_bits[assignment[node]] &= ~self.dependency_viewer.bits[node]',
    '            partition_users[id] = set(node.users)':
        '            partition_bits[id] = self.dependency_viewer.bits[node]\n            partition_users[id] = set(node.users)',
    '            partitions_by_id[id].add_node(node, node_order)':
        '            partitions_by_id[id].add_node(node, node_order)\n            partition_bits[id] |= self.dependency_viewer.bits[node]',
}
for before, after in replacements.items():
    assert source.count(before) == 1, before
    source = source.replace(before, after)
namespace = dict(vars(upstream))
exec(compile(source, '<bitset-partition-proposal>', 'exec'), namespace)


class BitsetPartitioner(partitioner.HabanaPartitioner):
    propose_partitions = namespace['propose_partitions']

    def __init__(self, graph_module, sup_op=partitioner.HabanaClusterOperatorSupport):
        self.graph_module = graph_module
        self.operator_support = sup_op()
        self.allows_single_node_partition = True
        self.non_compute_ops = []
        self.allowed_single_node_partition_ops = []
        self.skip_horizontal_fusion = False
        self.dependency_viewer = Dependencies(graph_module)


def install():
    partitioner.HabanaPartitioner = BitsetPartitioner
    passes.HabanaPartitioner = BitsetPartitioner


def validate():
    import operator
    import random
    import torch

    class Support(partitioner.HabanaClusterOperatorSupport):
        def is_node_supported(self, submodules, node):
            return node.op == 'call_function' and node.meta['placement'] == 'hpu_cluster'

    for seed in range(500):
        rng = random.Random(seed)
        graph = torch.fx.Graph()
        nodes = [graph.placeholder('x')]
        for _ in range(rng.randint(5, 80)):
            node = graph.call_function(rng.choice((operator.add, operator.mul)),
                                       (rng.choice(nodes), rng.choice(nodes)))
            node.meta['placement'] = 'hpu_cluster' if rng.random() < .7 else 'eager'
            nodes.append(node)
        graph.output(tuple(nodes[-4:]))
        module = torch.fx.GraphModule(torch.nn.Module(), graph)
        for vertical in (False, True):
            reference = upstream.CapabilityBasedPartitioner(module, Support(), allows_single_node_partition=True,
                                                            skip_horizontal_fusion=vertical)
            actual = BitsetPartitioner(module, Support)
            actual.skip_horizontal_fusion = vertical
            left = reference.propose_partitions()
            right = actual.propose_partitions()
            signature = lambda groups: {frozenset(node.name for node in group.nodes) for group in groups}
            assert signature(left) == signature(right), (seed, vertical)
    print('1000 exact partition comparisons passed', flush=True)


if __name__ == '__main__':
    validate()
