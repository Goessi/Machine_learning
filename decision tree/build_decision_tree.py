## 4. Determining the Column to Split On ##

def find_best_column(data, target_name, columns):
    igs = []
    for col in columns:
        ig = calc_information_gain(data,col, target_name)
        igs.append(ig)
    ind = igs.index(max(igs))
    column = columns[ind]
    return column

# A list of columns to potentially split income with
columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]
income_split = find_best_column(income, 'high_income', columns)

## 5. Creating a Simple Recursive Algorithm ##


label_1s = []
label_0s = []

def id3(data, target, columns):
    # The pandas.unique method will return a list of all the unique values in a series
    unique_targets = pandas.unique(data[target])
    
    if len(unique_targets) == 1:
        if unique_targets == 1:
            label_1s.append(unique_targets)
            return label_1s
        else:
            label_0s.append(unique_targets)
            return label_0s
    
    # Find the best column to split on in our data
    best_column = find_best_column(data, target, columns)
    # Find the median of the column
    column_median = data[best_column].median()
    
    # Create the two splits
    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]    
    # Loop through the splits and call id3 recursively
    for split in [left_split, right_split]:
        # Call id3 recursively to process each branch
        id3(split, target, columns)
    
# Create the data set that we used in the example on the last screen
data = pandas.DataFrame([
    [0,20,0],
    [0,60,2],
    [0,40,1],
    [1,25,1],
    [1,35,2],
    [1,55,1]
    ])
# Assign column names to the data
data.columns = ["high_income", "age", "marital_status"]

# Call the function on our data to set the counters properly
id3(data, "high_income", ["age", "marital_status"])

## 7. Storing the Tree ##

# Create a dictionary to hold the tree  
# It has to be outside of the function so we can access it later
tree = {}

# This list will let us number the nodes  
# It has to be a list so we can access it inside the function
nodes = []

def id3(data, target, columns, tree):
    unique_targets = pandas.unique(data[target])
    
    # Assign the number key to the node dictionary
    nodes.append(len(nodes) + 1)
    tree["number"] = nodes[-1]

    if len(unique_targets) == 1:
        if unique_targets == 1:
            tree['label'] = 1
        else:
            tree['label'] = 0
        return tree
    
    best_column = find_best_column(data, target, columns)
    column_median = data[best_column].median()
    
    tree['column'] = best_column
    tree['median'] = column_median
    
    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]
    split_dict = [["left", left_split], ["right", right_split]]
    
    for name, split in split_dict:
        tree[name] = {}
        id3(split, target, columns, tree[name])

# Call the function on our data to set the counters properly
id3(data, "high_income", ["age", "marital_status"], tree)

## 8. Printing Labels for a More Attractive Tree ##

def print_with_depth(string, depth):
    # Add space before a string
    prefix = "    " * depth
    # Print a string, and indent it appropriately
    print("{0}{1}".format(prefix, string))
    
    
def print_node(tree, depth):
    # Check for the presence of "label" in the tree
    if "label" in tree:
        # If found, then this is a leaf, so print it and return
        print_with_depth("Leaf: Label {0}".format(tree["label"]), depth)
        # This is critical -- without it, you'll get infinite recursion
        return
    # Print information about what the node is splitting on
    print_with_depth("{0} > {1}".format(tree["column"], tree["median"]), depth)
    
    # Create a list of tree branches
    branches = [tree["left"], tree["right"]]
        
    for bra in branches:
        print_node(bra, depth = depth + 1)

print_node(tree, 0)

## 10. Making Predictions Automatically ##

def predict(tree, row):
    if "label" in tree:
        return tree["label"]
    
    column = tree["column"]
    median = tree["median"]
    
    if row[column] <= median:
        pre = predict(tree['left'], row)
        return pre
    else:
        return predict(tree['right'], row)

# Print the prediction for the first row in our data
print(predict(tree, data.iloc[0]))

## 11. Making Multiple Predictions ##

new_data = pandas.DataFrame([
    [40,0],
    [20,2],
    [80,1],
    [15,1],
    [27,2],
    [38,1]
    ])
# Assign column names to the data
new_data.columns = ["age", "marital_status"]

def batch_predict(tree, df):
    pre = df.apply(lambda x: predict(tree, x), axis=1)
    return pre

predictions = batch_predict(tree, new_data)
