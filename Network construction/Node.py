class Node:
    def __init__(self, node_type, node_id, attributes=None):
        self.node_type = node_type #Reaction, Compound, Protein
        self.node_id = node_id
        self.attributes = {"node_id": node_id}

    def update_attributes(self, attribute_type, attribute):
        # Method to update node attributes
        if attribute_type in self.attributes.keys:
            self.attributes[attribute_type] = attribute
            print("Attribute:{attribute_type} is updated")
        else:
            self.attributes[attribute_type] = attribute
            print("New Attribute type {attribute_type} added")

    def get_all_attributes(self):
        # Method to get node attributes
        return self.attributes
    
    def get_specific_attributes(self, attribute):
        # Method to get node attributes
        return self.attributes[attribute]

    def __str__(self):
        # String representation of the node for easier debugging
        return f"{self.node_type} Node {self.node_id} with attributes {self.attributes}"