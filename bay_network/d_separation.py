from graphviz import Digraph


def dfs(dag, targets):
    pass


if __name__ == "__main__":
    dag_adj_list = {"A": ["C"], "B": ["C"], "C": ["D", "E"], "D": ["F"], "F": ["G"]}

    dfs(dag_adj_list, "A")
