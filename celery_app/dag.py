class Node:
    def __init__(self, name, func, kwargs, dependencies):
        self.name = name
        self.func = func
        self.kwargs = kwargs
        self.dependencies = dependencies
        self.inputs = {}
        self.outputs = {}

    def run(self, inputs):
        self.inputs = inputs
        self.outputs = self.func(**inputs)
        print(f"Executing {self.name} with inputs: {inputs}, outputs: {self.outputs}")


class StateDAG:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, name, func, kwargs, dependencies):
        self.nodes[name] = Node(name, func, kwargs, dependencies)
        for dep in dependencies:
            if dep not in self.edges:
                self.edges[dep] = []
            self.edges[dep].append(name)

    def run(self):
        sorted_nodes = self._topological_sort()
        print(f"Sorted Nodes: {sorted_nodes}")

        for node in sorted_nodes:
            self.nodes[node].run(self._get_inputs(node))
            print(f"Node {node} executed, outputs: {self.nodes[node].outputs}")

    def _topological_sort(self):
        sorted_nodes = []
        visited = set()

        def visit(node):
            if node not in visited:
                visited.add(node)
                if node in self.edges:
                    for neighbor in self.edges[node]:
                        visit(neighbor)
                sorted_nodes.insert(0, node) 

        for node in self.nodes:
            visit(node)

        return sorted_nodes

    def _get_inputs(self, node):
        inputs = {}
        for dep in self.nodes[node].dependencies:
            inputs.update(self.nodes[dep].outputs.copy())
        return inputs

    def __repr__(self):
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"


def func1(**kwargs):
    print("Executing func1")
    return {'output1': 42}


def func2(**kwargs):
    print("Executing func2")
    return {'output2': kwargs['output1'] * 2}

def func3(**kwargs):
    print("Executing func3")
    return {'output3': 10}


def func4(**kwargs):
    print("Executing func4")
    return {'output4': kwargs['output2'] + kwargs['output3']}


if __name__ == '__main__':
    dag = StateDAG()
    dag.add_node(name='node3', func=func3, kwargs={}, dependencies=[])
    dag.add_node(name='node4', func=func4, kwargs={}, dependencies=['node2', 'node3'])
    dag.add_node(name='node1', func=func1, kwargs={}, dependencies=[])
    dag.add_node(name='node2', func=func2, kwargs={}, dependencies=['node1'])

    dag.run()