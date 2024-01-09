
from Node import Node

class Edge:
    def __init__(self, edge_type, edge_id, source, target, attributes=None):
        self.edge_type = edge_type #Drug, Reaction, Compound, Protein
        self.edge_id = edge_id
        self.source = source
        self.target = target
        self.attributes = {"node_id": edge_id}

    def update_attributes(self, attribute_type, attribute):
        # Method to update node attributes
        if attribute_type in self.attributes.keys:
            self.attributes[attribute_type] = attribute
            print("Attribute:{attribute_type} is updated")
        else:
            self.attributes[attribute_type] = attribute
            print("New Attribute type {attribute_type} added")

    def update_source_target(self, tuple):
        source = tuple[0]
        target = tuple[1]

        self.source = source
        self.target = target

        self.source_id = source.node_id
        self.target_id = target.node_id

    def get_relations(self):

        return (self.source_id, self.target_id)

    def get_all_attributes(self):
        # Method to get node attributes
        return self.attributes
    
    def get_specific_attributes(self, attribute):
        # Method to get node attributes
        return self.attributes[attribute]

    def display(self):
        # String representation of the node for easier debugging
        a = f"{self.edge_type} Edge {self.edge_id} between ({self.source},{self.target}) with attributes {self.attributes}"
        return str(a)

