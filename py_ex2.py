from collections import Counter
import math

ALGORITHMS = ["Decision tree", "KNN", "Naive base"]
KNN_K_CONSTANT = 5
OUTPUT_FILE_PATH = "output.txt"


def create_table_from_file(table_path):
    """
    This function creates the table for our program.
    Each line is a pair of (data entry, class_name),
    Where data entry is a dictionary of attributes names and values.
    Line Example: ([[pclass, 2nd], [age, adult], [sex, male]], no)
    :param table_path: path of table file
    :return: table as described
    """
    table = []
    with open(table_path, "r") as table_file:
        attributes_names = table_file.readline().split()
        for line in table_file.readlines():
            line_dictionary = dict(zip(attributes_names, line.split()))
            _, class_name = line_dictionary.popitem()
            table.append((line_dictionary, class_name))
    return table


def get_classes_names(table):
    """
    Extracts class names from table
    :param table: table of data entries and classification
    :return all classes names on the table
    """
    return set(map(lambda x: x[1], table))


def get_class_probability(table, class_name):
    """
    This function counts the number of data entries on this class, divided by total data entries count, giving class
    predicted probability
    :param table: table of data entries and classification
    :param class_name: name of current class
    :return: p(class_name)
    """
    return len(list(filter(lambda x: x[1] == class_name, table))) / float(len(table))


def get_attributes_values(table, attribute_name):
    """
    This function extracts the attributes possible values from the table
    :param table: table of data entries and classification
    :param attribute_name: name of current attribute
    :return: set of all attribute's possible values
    """
    return set(map(lambda x: x[0][attribute_name], table))


def calc_hamming_distance(data_entry_a, data_entry_b):
    """
    Counts how many times there is a difference between attributes values between the two data entries
    :param data_entry_a: first data entry
    :param data_entry_b: second data entry
    :return: hamming distance
    """
    hamming_distance = 0
    for attribute_name in data_entry_a.keys():
        if data_entry_a[attribute_name] != data_entry_b[attribute_name]:
            hamming_distance += 1
    return hamming_distance


def naive_bayes(train_table, data_entry):
    """
    Classifies the data_entry according the Naive-Bayes algorithm
    :param train_table: table of data entries and classification on train sample
    :param data_entry: a vector of attributes, as described above
    :return: the data_entry predicted class
    """
    # This variable will contain dictionary of class names to calculated smoothed probabilities
    class_probabilities = {}

    # Iterate over all class names
    for class_name in get_classes_names(train_table):
        # Set class probability to p(class_name) to start the calculation
        class_probability = get_class_probability(train_table, class_name)

        # Foreach attribute name & value on data entry
        for attribute_name, attribute_value in data_entry.items():
            # Count the number of potential values of current attribute
            attributes_values_count = len(get_attributes_values(train_table, attribute_name))

            # Count the number of data_entries on train_table which has the same classification as current class name
            classified_data_entries_count = len(list(filter(lambda x: x[1] == class_name, train_table)))

            # count the number of data_entries on train_table which has the same classification and attribute value
            matching_and_classified_data_entries_count = len(list(filter(lambda x: x[1] == class_name and (x[0][attribute_name] == attribute_value), train_table)))

            # Calc attribute smoothed probability
            attribute_probability = (float(matching_and_classified_data_entries_count + 1) / float(classified_data_entries_count + attributes_values_count))

            # Multiply all attributes smoothed probabilities and p(class_name)
            class_probability *= attribute_probability

        # Fill Dictionary with classes names and probabilities
        class_probabilities[class_name] = class_probability

    # If all classes has the same probability, return class with maximum probability in train table
    if len(set(class_probabilities.values())) == 1:
        return max(get_classes_names(train_table), key=lambda x: get_class_probability(train_table, x))

    # Return the class with the biggest probability
    return max(class_probabilities, key=class_probabilities.get)


def knn(train_table, data_entry):
    """
    Classifies the data_entry according the K-Nearest-Neighbors algorithm
    :param train_table: table of data entries and classification on train sample
    :param data_entry: a vector of attributes, as described above
    :return: the data_entry predicted class
    """
    k_nearest_neighbors = []
    index = 0
    for line in train_table:
        # Get Hamming-Distance from current data_entry on train table to the received dat entry
        hamming_distance = calc_hamming_distance(line[0], data_entry)

        # Insert to list. Hamming-Distance & index are used to sort the list with distances & order of appearance
        k_nearest_neighbors.append([hamming_distance, index, line[1]])

        index += 1

    # Sort list - in case of equal distances, sort by index (order of appearance)
    k_nearest_neighbors.sort()

    # Get only classes names list
    k_nearest_neighbors = list(map(lambda x: x[2], k_nearest_neighbors))

    # Get top K neighbors on list
    k_nearest_neighbors = k_nearest_neighbors[:KNN_K_CONSTANT]

    # Count each class name appearances
    knn_counter = Counter(k_nearest_neighbors)

    # Return class name with maximum appearances
    return max(k_nearest_neighbors, key=knn_counter.get)


class DecisionTreeNode(object):
    """
    Defines node on the DecisionTree
    """
    def __init__(self, class_name=None):
        self.attribute_name = None
        self.attribute_value = None
        self.children = []  # List of child nodes
        self.class_name = class_name

    def __str__(self):
        string = self.attribute_name + "=" + self.attribute_value
        if self.class_name:
            string += ":" + self.class_name
        return string


class DecisionTree(object):
    TREE_OUTPUT_FILE = "output_tree.txt"

    def __init__(self, table):
        self.table = table  # Table of Data Entries
        self.attributes_names = list(self.table[0][0])  # Extracts the attribute names from the first line

        # Run rhe recursive algorithm
        self.root = self.build_tree(self.table, self.attributes_names, self.max_probability_class(self.table))

        # Write the result to TREE_OUTPUT_FILE
        with open(self.TREE_OUTPUT_FILE, "w") as save_file:
            save_file.write(str(self))

    def __str__(self):
        lines = []
        nodes = []

        # Assign 0 depth for all nodes
        for child_node in self.root.children:
            nodes.append([child_node, 0])

        # Work down the tree, developing all nodes
        while len(nodes) != 0:
            node, depth = nodes[0]
            del nodes[0]  # Remove current node from list
            prefix = "\t" * depth + ("|" if depth > 0 else "")
            lines.append(prefix + str(node))

            # Add all nodes children to the top of the list for development, with extra depth
            for child_node in reversed(node.children):
                nodes.insert(0, [child_node, depth + 1])
        return "\n".join(lines)

    @staticmethod
    def entropy(table):
        """
        Calculates Entropy according to its formula
        :param table: table of data entries and classification
        :return: Entropy value
        """
        entropy = 0
        for class_name in get_classes_names(table):
            class_probability = get_class_probability(table, class_name)
            entropy -= class_probability * math.log(class_probability)
        return entropy

    @staticmethod
    def info_gain(table, attribute_name):
        """
        Calcs Information Gain of specific attribute
        :param table: table of data entries and classification
        :param attribute_name: attribute's name
        :return: Information Gain value
        """
        children_tables = []

        # Iterate over all attribute values and append all data entries where attribute value is matching current value
        for attribute_value in get_attributes_values(table, attribute_name):
            children_tables.append(list(filter(lambda x: x[0][attribute_name] == attribute_value, table)))

        # Calc Entropy sum, weighted by probabilities of current attribute value
        children_entropy_sum = sum(map(lambda x: (len(x) / len(table)) * DecisionTree.entropy(x), children_tables))

        # Return main table entropy - entropy sum calculated
        return DecisionTree.entropy(table) - children_entropy_sum

    @staticmethod
    def max_probability_class(table):
        """
        :param table: table of data entries and classification
        :return: Name of the class with the largest probability
        """
        return max(get_classes_names(table), key=lambda x: get_class_probability(table, x))

    def build_tree(self, table, attributes_names, default):
        """
        MAIN LOGIC - run the recursive algorithm
        :param table: table of data entries and classification
        :param attributes_names: set of all attribute names in the table
        :param default: class_name with largest probability
        :return: DecisionTreeNode which is the root of the appropriate DecisionTree
        """
        if len(table) == 0:  # No more data entries in the table
            return DecisionTreeNode(default)
        if set(map(lambda x: x[1], table)) == 1:  # All remaining lines contains the same classification
            node = DecisionTreeNode()
            node.class_name = table[0][1]
            return node
        if len(attributes_names) == 0:  # No more attributes
            node = DecisionTreeNode()
            node.class_name = DecisionTree.max_probability_class(table)
            return node

        node = DecisionTreeNode()

        # Choose attribute with maximum Information Gain
        attribute_name = max(attributes_names, key=lambda x: DecisionTree.info_gain(table, x))

        # Iterate over all value of selected attribute
        for attribute_value in get_attributes_values(self.table, attribute_name):
            # Derive sub-table where current attribute value appears
            child_sub_table = list(filter(lambda x: x[0][attribute_name] == attribute_value, table))
            child_attributes_names = attributes_names[:]
            child_attributes_names.remove(attribute_name)

            # Activate the recursive algorithm using child-data derived when choosing current attribute value
            child_node = self.build_tree(child_sub_table, child_attributes_names, DecisionTree.max_probability_class(table))

            # Assign values to child root node and append as child to main node
            child_node.attribute_name = attribute_name
            child_node.attribute_value = attribute_value
            node.children.append(child_node)

        # Sort Children before returning root node
        node.children = sorted(node.children, key=lambda x: x.attribute_value)
        return node

    def classify(self, data_entry):
        """
        Use the DecisionTree to determine the classification of new data
        :param data_entry: table data entry - dictionary pf attributes names and value on specific table line
               Structure example: [[pclass, 2nd], [age, adult], [sex, male]]
        :return: class name for the data entry
        """
        node = self.root

        # Start from root and work down using the values in the data entry until you find classification on leaf node
        while not node.class_name:
            for child in node.children:
                if data_entry[child.attribute_name] == child.attribute_value:
                    node = child
                    break
        return node.class_name


def compute_accuracy(prediction, actual_result):
    """
    :param prediction: classification predicted by algorithm
    :param actual_result: classification given as actual result
    :return: accuracy index
    """
    # Error, can't calc accuracy when lengths don't match
    if len(prediction) != len(actual_result):
        return 0.0

    # Count all class names where there is a match between prediction and actual result, divided by prediction length
    index = 0
    match_sum = 0
    while index != len(prediction):
        if prediction[index] == actual_result[index]:
            match_sum += 1
        index += 1
    return round((float(match_sum) / float(len(prediction))) + 0.005, 2)


def create_prediction(train_table, test_data_entries, algorithm):
    """
    Creates prediction of test data vector using the data & algorithm
    :param train_table: the training table
    :param test_data_entries: all data entries on test table
    :param algorithm: desired algorithm
    :return: prediction - a list of of predicted classifications for each data entry on test_data_entries, using the DecisionTree's classify func
    """
    if algorithm == "Decision tree":
        decision_tree = DecisionTree(train_table)

        # Activate Classify on each test data entry
        return list(map(lambda x: decision_tree.classify(x), test_data_entries))
    if algorithm == "KNN":
        return list(map(lambda x: knn(train_table, x), test_data_entries))
    if algorithm == "Naive base":
        return list(map(lambda x: naive_bayes(train_table, x), test_data_entries))


def build_output_file(train_table, test_table):
    """
    Uses the train & test tables to construct output file
    :param train_table: the training table for the algorithms
    :param test_table: the test table
    :return:
    """
    first_line = "\t".join(["Num", "DT ", "KNN", "naiveBase"])

    # Separate test_table to data entries & classifications
    test_data_entries = list(map(lambda x: x[0], test_table))
    actual_results = list(map(lambda x: x[1], test_table))

    # Create prediction vector for each algorithm
    algorithms_predictions = list(map(lambda x: create_prediction(train_table, test_data_entries, x), ALGORITHMS))

    # Calc algorithms accuracies
    algorithms_accuracies = []
    for i in range(len(algorithms_predictions)):
        algorithms_accuracies.append(str(compute_accuracy(algorithms_predictions[i], actual_results)))

    # Write results to file
    with open(OUTPUT_FILE_PATH, "w") as output_file:
        output_file.write(first_line + "\n")
        for i in range(len(actual_results)):
            output_file.write("%d\t" % (i + 1))
            for prediction in algorithms_predictions:
                output_file.write(prediction[i] + "\t")
            output_file.write("\n")

        output_file.write("\t" + "\t".join(algorithms_accuracies))


def main():
    train_table = create_table_from_file("train.txt")
    test_table = create_table_from_file("test.txt")
    build_output_file(train_table, test_table)


if __name__ == "__main__":
    main()

