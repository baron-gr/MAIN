### Binary Search Trees, Traversals, and Balancing

## Classes and constructors
# class User:
#     def __init__(self, username, name, email):
#         self.username = username
#         self.name = name
#         self.email = email
        
#     def intro(self, guest_name):
#         print("Hi {}, I'm {}! Contact me at {} .".format(guest_name, self.name, self.email))
    
# user2 = User('john_d','John Doe', 'john@example.com')
# user3 = User('jane','Jane Doe', 'jane@example.com')

# user3.intro('David')

## Repr and Str method
import time


class User:
    def __init__(self, username, name, email):
        self.username = username
        self.name = name
        self.email = email
    
    def __repr__(self):
        return "User(username='{}', name='{}', email='{}')".format(self.username, self.name, self.email)
    def __str__(self):
        return self.__repr__()

# user4 = User('jane','Jane Doe', 'jane@example.com')
# print(user4)

## Database Class
class UserDatabase:
    def insert(self, user):
        pass
    def find(self, username):
        pass
    def update(self, user):
        pass
    def list_all(self):
        pass

john1 = User('john123', 'john A', 'john1@example.com')
john2 = User('john223', 'john B', 'john2@example.com')
john3 = User('john323', 'john C', 'john3@example.com')
john4 = User('john423', 'john D', 'john4@example.com')
john5 = User('john523', 'john E', 'john5@example.com')
john6 = User('john623', 'john F', 'john6@example.com')

users = [john1, john2, john3, john4, john5]

class UserDatabase:
    def __init__(self):
        self.users = []
    
    def insert(self, user):
        i = 0
        while i < len(self.users):
            if self.users[i].username > user.username:
                break
            i += 1
        self.users.insert(i, user)
    
    def find(self, username):
        for user in self.users:
            if user.username == username:
                return user
    
    def update(self, user):
        target = self.find(user.username)
        target.name, target.email = user.name, user.email
    
    def list_all(self):
        return self.users

database = UserDatabase()

database.insert(john1)
database.insert(john2)
database.insert(john3)

# user = database.find('john123')
# print(user)

database.update(User(username='john123', name='john Ab', email='john@example.com'))
user = database.find('john123')
# print(user)

database.insert(john4)
# print(database.list_all())

## Complexity analysis
## Insert: O(N), Find: O(N), Update: O(N), List: O(1)

## Balanced Binary Search Trees
## Tree of height k, with N nodes -> k = log(N) + 1

## Implementing a simple binary tree
class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

node0 = TreeNode(3)
node1 = TreeNode(4)
node2 = TreeNode(5)

node0.left = node1
node0.right = node2

tree = node0
# print(tree.key)
# print(tree.left.key)
# print(tree.right.key)

node0 = TreeNode(2)
node1 = TreeNode(3)
node2 = TreeNode(1)
node3 = TreeNode(5)
node4 = TreeNode(3)
node5 = TreeNode(7)
node6 = TreeNode(4)
node7 = TreeNode(6)
node8 = TreeNode(8)

node0.left = node1
node1.right = node3
node1.left = node2
node3.left = node4
node3.right = node5
node4.right = node6
node5.left = node7
node5.right = node8

tree = node0

# print(tree.key)

## Tree builder
tree_tuple = ((1,3,None),2,((None,3,4),5,(6,7,8)))

def parse_tuple(data):
    # print(data)
    if isinstance(data, tuple) and len(data) == 3:
        node = TreeNode(data[1])
        node.left = parse_tuple(data[0])
        node.right = parse_tuple(data[2])
    elif data is None:
        node = None
    else:
        node = TreeNode(data)
    return node

tree2 = parse_tuple(tree_tuple)
# print(tree2.key)
# print(tree2.left.key,tree2.right.key)
# print(tree2.left.left.key,tree2.left.right.key,tree2.right.left.key,tree2.right.right.key)

## Tuple builder
# def tree_to_tuples(node):
    # 

## Display tree
def display_keys(node, space='\t', level=0):
    if node is None:
        print(space*level + '/')
        return
    if node.left is None and node.right is None:
        print(space*level + str(node.key))
        return
    display_keys(node.right, space, level+1)
    print(space*level + str(node.key))
    display_keys(node.left, space, level+1)

display_keys(tree2, '       ')

## Traversing a Binary Tree

## Inorder Traversal: Traverse left subtree recursively inorder
## i.e. Go to left most node, then follow branches, always staying against inner edge. 

## Preorder Traversal: Traverse current node, then traverse left subtree recursively preorder
## Start at root node, then follow left mode node

## Postorder Traversal: Traverse