#!/usr/bin/env python3

import itertools
from collections import defaultdict

import numpy as np
from networkx.algorithms import bipartite

from pgmpy.models.MarkovModel import MarkovModel
from pgmpy.base import UndirectedGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors import factor_product


class FactorGraph(UndirectedGraph):
    """
    Class for representing factor graph.

    DiscreteFactor graph is a bipartite graph representing factorization of a function.
    They allow efficient computation of marginal distributions through sum-product
    algorithm.

    A factor graph contains two types of nodes. One type corresponds to random
    variables whereas the second type corresponds to factors over these variables.
    The graph only contains edges between variables and factor nodes. Each factor
    node is associated with one factor whose scope is the set of variables that
    are its neighbors.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is
        created. The data is an edge list.

    Examples
    --------
    Create an empty FactorGraph with no nodes and no edges

    >>> from pgmpy.models import FactorGraph
    >>> G = FactorGraph()

    G can be grown by adding variable nodes as well as factor nodes

    **Nodes:**

    Add a node at a time or a list of nodes.

    >>> G.add_node('a')
    >>> G.add_nodes_from(['a', 'b'])
    >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
    >>> G.add_factors(phi1)
    >>> G.add_nodes_from([phi1])

    **Edges:**

    G can also be grown by adding edges.

    >>> G.add_edge('a', phi1)

    or a list of edges

    >>> G.add_edges_from([('a', phi1), ('b', phi1)])
    """

    def __init__(self, ebunch=None):
        super(FactorGraph, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.factors = []

    def add_edge(self, u, v, **kwargs):
        """
        Add an edge between variable_node and factor_node.

        Parameters
        ----------
        u, v: nodes
            Nodes can be any hashable Python object.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> G.add_nodes_from([phi1])
        >>> G.add_edge('a', phi1)
        """
        if u != v:
            super(FactorGraph, self).add_edge(u, v, **kwargs)
        else:
            raise ValueError("Self loops are not allowed")

    def add_factors(self, *factors, replace=False):
        """
        Associate a factor to the graph.
        See factors class for the order of potential values.

        Parameters
        ----------
        *factor: pgmpy.factors.DiscreteFactor object
            A factor object on any subset of the variables of the model which
            is to be associated with the model.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> G.add_edges_from([('a', phi1), ('b', phi1),
        ...                   ('b', phi2), ('c', phi2)])
        """
        for factor in factors:
            if set(factor.variables) - set(factor.variables).intersection(
                set(self.nodes())
            ):
                raise ValueError(
                    "Factors defined on variable not in the model", factor.__repr__()
                )

            if replace:
                for fa in self.factors:
                    if set(factor.variables) == set(fa.variables):
                        neighbors = self.neighbors(fa)
                        self.remove_factors(fa)
                        self.add_node(factor)
                        self.add_edges_from([(factor, neigh) for neigh in neighbors])
                self.factors.append(factor)

            else:
                self.factors.append(factor)

    def remove_factors(self, *factors):
        """
        Removes the given factors from the added factors.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1)
        >>> G.remove_factors(phi1)
        """
        for factor in factors:
            self.factors.remove(factor)
            # If factor is also in the graph, remove the node and corresponding edges.
            if factor in self.nodes:
                self.remove_node(factor)

    def get_cardinality(self, node=None):
        """
        Returns the cardinality of the node

        Parameters
        ----------
        node: any hashable python object (optional)
            The node whose cardinality we want. If node is not specified returns a
            dictionary with the given variable as keys and their respective cardinality
            as values.

        Returns
        -------
        int or dict : If node is specified returns the cardinality of the node.
                      If node is not specified returns a dictionary with the given
                      variable as keys and their respective cardinality as values.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_edges_from([('a', phi1), ('b', phi1),
        ...                   ('b', phi2), ('c', phi2)])
        >>> G.add_factors(phi1, phi2)
        >>> G.get_cardinality()
        defaultdict(<class 'int'>, {'c': 2, 'b': 2, 'a': 2})

        >>> G.get_cardinality('a')
        2
        """
        if node:
            for factor in self.factors:
                for variable, cardinality in zip(factor.scope(), factor.cardinality):
                    if node == variable:
                        return cardinality
        else:
            cardinalities = defaultdict(int)
            for factor in self.factors:
                for variable, cardinality in zip(factor.scope(), factor.cardinality):
                    cardinalities[variable] = cardinality
            return cardinalities

    def check_model(self):
        """
        Check the model for various errors. In the same time it also updates 
        the cardinalities of all the random variables.
        """
        factor_nodes = set(
            [phi for phi in self.nodes if isinstance(phi, DiscreteFactor)]
        )
        variables = set(self.nodes) - factor_nodes

        if not variables:
            raise ValueError("The factor graph does not contain any variable.")
        if not self.factors:
            raise ValueError("The factor graph does not contain any factor.")
        if len(factor_nodes) != len(self.factors):
            raise ValueError("Factors not associated with all the factor nodes.")

        variables_in_factor_scopes = set(
            [v for phi in self.factors for v in phi.scope()]
        )
        nodes = list(self.nodes)

        # Check variables and factor scopes overlap
        if not variables == variables_in_factor_scopes:
            raise ValueError("Variables and factor's scopes are not the same set.")
        # all variables and factors are also nodes
        if not all(v in nodes for v in variables):
            raise ValueError("Not all variables are nodes in the graph.")
        if not all(phi in nodes for phi in self.factors):
            raise ValueError("Not all factors are nodes in the graph.")
        # For all edges: e[0] in variables and e[1] in self.factors or vice-versa (i.e. is bipartite)
        if not all(
            (e[0] in variables and e[1] in self.factors)
            or (e[1] in variables and e[0] in self.factors)
            for e in self.edges
        ):
            raise ValueError("The factor graph is not bipartite.")

        cardinalities = self.get_cardinality()
        if len(variables) != len(cardinalities):
            raise ValueError("Factors for all the variables not defined")

        for factor in self.factors:
            for variable, cardinality in zip(factor.scope(), factor.cardinality):
                if cardinalities[variable] != cardinality:
                    raise ValueError(
                        f"Cardinality of variable {variable} not matching among factors"
                    )

        return True

    def get_variable_nodes(self):
        """
        Returns variable nodes present in the graph.

        Before calling this method make sure that all the factors are added
        properly.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_factors(phi1, phi2)
        >>> G.add_edges_from([('a', phi1), ('b', phi1),
        ...                   ('b', phi2), ('c', phi2)])
        >>> G.get_variable_nodes()
        ['a', 'c', 'b']
        """
        self.check_model()

        variable_nodes = set([x for factor in self.factors for x in factor.scope()])
        return list(variable_nodes)

    def get_factor_nodes(self):
        """
        Returns factors nodes present in the graph.

        Before calling this method make sure that all the factors are added
        properly.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_factors(phi1, phi2)
        >>> G.add_edges_from([('a', phi1), ('b', phi1),
        ...                   ('b', phi2), ('c', phi2)])
        >>> G.get_factor_nodes()
        [<DiscreteFactor representing phi(b:2, c:2) at 0x4b8c7f0>,
         <DiscreteFactor representing phi(a:2, b:2) at 0x4b8c5b0>]
        """
        self.check_model()

        variable_nodes = self.get_variable_nodes()
        factor_nodes = set(self.nodes()) - set(variable_nodes)
        return list(factor_nodes)

    def to_markov_model(self):
        """
        Converts the factor graph into markov model.

        A markov model contains nodes as random variables and edge between
        two nodes imply interaction between them.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_edges_from([('a', phi1), ('b', phi1),
        ...                   ('b', phi2), ('c', phi2)])
        >>> mm = G.to_markov_model()
        """
        mm = MarkovModel()

        variable_nodes = self.get_variable_nodes()

        if len(set(self.nodes()) - set(variable_nodes)) != len(self.factors):
            raise ValueError("Factors not associated with all the factor nodes.")

        mm.add_nodes_from(variable_nodes)
        for factor in self.factors:
            scope = factor.scope()
            mm.add_edges_from(itertools.combinations(scope, 2))
            mm.add_factors(factor)

        return mm

    def to_junction_tree(self):
        """
        Create a junction treeo (or clique tree) for a given factor graph.

        For a given factor graph (H) a junction tree (G) is a graph
        1. where each node in G corresponds to a maximal clique in H
        2. each sepset in G separates the variables strictly on one side of
        edge to other

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_edges_from([('a', phi1), ('b', phi1),
        ...                   ('b', phi2), ('c', phi2)])
        >>> mm = G.to_markov_model()
        """
        mm = self.to_markov_model()
        return mm.to_junction_tree()

    def get_factors(self, node=None):
        """
        Returns the factors that have been added till now to the graph.

        If node is not None, it would return the factor corresponding to the
        given node.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_edges_from([('a', phi1), ('b', phi1),
        ...                   ('b', phi2), ('c', phi2)])
        >>> G.get_factors()
        >>> G.get_factors(node=phi1)
        """
        if node is None:
            return self.factors
        else:
            factor_nodes = self.get_factor_nodes()
            if node not in factor_nodes:
                raise ValueError(
                    "Factors are not associated with the " "corresponding node."
                )
            factors = list(
                filter(
                    lambda x: set(x.scope()) == set(self.neighbors(node)), self.factors
                )
            )
            return factors[0]

    def get_partition_function(self):
        """
        Returns the partition function for a given undirected graph.

        A partition function is defined as

        .. math:: \sum_{X}(\prod_{i=1}^{m} \phi_i)

        where m is the number of factors present in the graph
        and X are all the random variables present.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_edges_from([('a', phi1), ('b', phi1),
        ...                   ('b', phi2), ('c', phi2)])
        >>> G.get_factors()
        >>> G.get_partition_function()
        """
        factor = self.factors[0]
        factor = factor_product(
            factor, *[self.factors[i] for i in range(1, len(self.factors))]
        )
        if set(factor.scope()) != set(self.get_variable_nodes()):
            raise ValueError("DiscreteFactor for all the random variables not defined.")

        return np.sum(factor.values)

    def copy(self):
        """
        Returns a copy of the model.

        Returns
        -------
        FactorGraph : Copy of FactorGraph

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_edges_from([('a', phi1), ('b', phi1),
        ...                   ('b', phi2), ('c', phi2)])
        >>> G_copy = G.copy()
        >>> G_copy.nodes()
        NodeView((<Factor representing phi(b:2, c:2) at 0xb4badd4c>, 'b', 'c',
          'a', <Factor representing phi(a:2, b:2) at 0xb4badf2c>))

        """
        copy = FactorGraph(self.edges())
        copy.add_nodes_from(self.nodes())

        if self.factors:
            factors_copy = [factor.copy() for factor in self.factors]
            copy.add_factors(*factors_copy)

        return copy

    @property
    def parameters(self):
        """Returns all the factors parameters as vector.

        Returns:
            NDArray
        """
        return np.concatenate([factor.values.flatten() for factor in self.factors])

    @parameters.setter
    def parameters(self, values) -> None:
        """Set the factors parameters."""
        sizes = [np.product(factor.cardinality) for factor in self.factors]
        values_ = np.split(values, np.cumsum(sizes)[:-1])
        for i, factor in enumerate(self.get_factors()):
            factor.values = values_[i].reshape(factor.cardinality)